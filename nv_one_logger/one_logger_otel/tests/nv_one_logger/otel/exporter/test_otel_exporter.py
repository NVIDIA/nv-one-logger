# SPDX-License-Identifier: Apache-2.0
"""Test suite for the OTelExporter class."""

import time
from typing import Generator
from unittest.mock import MagicMock

import pytest
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import (
    ErrorEvent,
    Event,
    StandardEventAttributeName,
    StandardEventName,
    TelemetryDataError,
)
from nv_one_logger.core.span import Span, StandardSpanAttributeName
from nv_one_logger.core.time import TracingTimestamp
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

from nv_one_logger.otel.exporter.otel_exporter import OTelExporter


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Create and return an InMemorySpanExporter instance."""
    return InMemorySpanExporter()


@pytest.fixture
def exporter(span_exporter: InMemorySpanExporter) -> Generator[OTelExporter, None, None]:
    """Create and initialize an OTelExporter instance."""
    exporter = OTelExporter(span_exporter)
    assert not exporter.ready
    exporter.initialize()
    assert exporter.ready

    yield exporter

    # clean up
    exporter.close()
    assert exporter.closed


def to_nano(time_sec: float) -> int:
    """Convert a time in seconds to nanoseconds."""
    return int(time_sec * 1_000_000_000)


def advance_time(time: TracingTimestamp, seconds: float) -> TracingTimestamp:
    """Advance a timestamp by the specified number of seconds."""
    return TracingTimestamp.for_timestamp(timestamp_sec=time.seconds_since_epoch + seconds, perf_counter=time.perf_counter_seconds + seconds)


class TestOTelExporter:
    """Test suite for the OTelExporter class."""

    def test_convert_timestamp(self, exporter: OTelExporter) -> None:
        """Test that timestamps are correctly converted to nanoseconds."""
        # Test with a specific timestamp
        time_sec = 45253523993243
        timestamp = TracingTimestamp.for_timestamp(timestamp_sec=time_sec, perf_counter=100, validate_timestamp=False)

        assert exporter._convert_timestamp(timestamp) == int(time_sec * 1_000_000_000)  # type: ignore[reportPrivateUsage]

    def test_convert_attributes(self, exporter: OTelExporter) -> None:
        """Test that attributes are correctly converted to OTEL format."""
        # Test with various attribute types
        timestamp = TracingTimestamp.for_timestamp(timestamp_sec=200, perf_counter=100, validate_timestamp=False)
        attributes = Attributes(
            {
                "string": "value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "list": [1, 2, 3],
                "unsupported dict type": {"key": "value"},
                "unsupported timestamp type": timestamp,
            }  # type: ignore
        )

        result = exporter._convert_attributes(attributes)  # type: ignore[reportPrivateUsage]

        # Check that supported types are preserved
        assert result
        assert result["string"] == "value"
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["list"] == [1, 2, 3]

        # Check that unsupported types are converted to string
        assert result["unsupported dict type"] == "{'key': 'value'}"
        assert result["unsupported timestamp type"] == "TracingTimestamp(_seconds_since_epoch=200, _perf_counter_seconds=100)"

    def test_export_span_without_attributes(self, exporter: OTelExporter, span_exporter: InMemorySpanExporter) -> None:
        """Test that a span without attributes is correctly exported."""
        # Create a test span
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=time.time(), perf_counter=100)
        test_span = Span.create(
            name="test_span1",
            start_time=start_time,
        )
        exporter.export_start(test_span)

        stop_time = advance_time(start_time, 5.31)
        test_span.stop(stop_time=stop_time)
        exporter.export_stop(test_span)
        exporter.close()

        # Verify that the span was created with correct parameters
        exported_spans = span_exporter.get_finished_spans()
        assert len(exported_spans) == 3

        # Fake span for start event
        fake_start_span = exported_spans[0]
        assert fake_start_span.name == "span_start_event_for_test_span1"
        assert fake_start_span.attributes == {StandardEventAttributeName.TIMESTAMP_MSEC: start_time.milliseconds_since_epoch}
        assert fake_start_span.start_time == to_nano(start_time.seconds_since_epoch)
        assert fake_start_span.end_time == to_nano(start_time.seconds_since_epoch)
        assert fake_start_span.context is not None
        assert fake_start_span.status.is_ok
        assert len(fake_start_span.events) == 0

        # Fake span for stop event
        fake_stop_span = exported_spans[1]
        assert fake_stop_span.name == "span_stop_event_for_test_span1"
        assert fake_stop_span.attributes == {StandardEventAttributeName.TIMESTAMP_MSEC: stop_time.milliseconds_since_epoch}
        assert fake_stop_span.start_time == to_nano(stop_time.seconds_since_epoch)
        assert fake_stop_span.end_time == to_nano(stop_time.seconds_since_epoch)
        assert fake_stop_span.context is not None
        assert fake_stop_span.status.is_ok
        assert len(fake_stop_span.events) == 0

        # Real span for stop event
        exported_span = exported_spans[2]
        assert exported_span.name == "test_span1"
        # The onelogger span id and duration attributes are automatically added
        assert exported_span.attributes == {
            "one_logger_span_id": test_span.id.int,
            StandardSpanAttributeName.DURATION_MSEC.value: 5310,
        }
        assert exported_span.parent is None
        assert exported_span.context is not None
        assert exported_span.status.is_ok
        assert len(exported_span.events) == 2
        start_event = exported_span.events[0]
        assert start_event.name == "span_start"
        assert start_event.attributes == fake_start_span.attributes
        assert start_event.timestamp == to_nano(start_time.seconds_since_epoch)
        stop_event = exported_span.events[1]
        assert stop_event.name == "span_stop"
        assert stop_event.attributes == fake_stop_span.attributes
        assert stop_event.timestamp == to_nano(stop_time.seconds_since_epoch)

        # Make sure the relationship between the real span and the fake ones is set up correctly
        assert fake_start_span.parent is not None
        assert fake_start_span.parent.span_id == exported_span.context.span_id
        assert fake_stop_span.parent is not None
        assert fake_stop_span.parent.span_id == exported_span.context.span_id

    def test_export_span_with_attributes(self, exporter: OTelExporter, span_exporter: InMemorySpanExporter) -> None:
        """Test that a span with attributes and its start/stop events are correctly exported."""
        # Create a test span
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=time.time(), perf_counter=100)
        test_span = Span.create(
            name="test_span2",
            span_attributes=Attributes({"span_attrib_key": "span_attrib_val"}),
            start_event_attributes=Attributes(
                {StandardEventAttributeName.TIMESTAMP_MSEC: start_time.milliseconds_since_epoch, "start_event_attr_key": "start_event_attr_val"}
            ),
            start_time=start_time,
        )
        exporter.export_start(test_span)

        stop_time = advance_time(start_time, 5.31)
        test_span.stop(stop_time=stop_time, stop_event_attributes=Attributes({"stop_event_attr_key": "stop_event_attr_val"}))
        exporter.export_stop(test_span)
        exporter.close()

        # Verify that the span was created with correct parameters
        exported_spans = span_exporter.get_finished_spans()
        assert len(exported_spans) == 3

        # Fake span for start event
        fake_start_span = exported_spans[0]
        assert fake_start_span.name == "span_start_event_for_test_span2"
        assert fake_start_span.attributes == {
            StandardEventAttributeName.TIMESTAMP_MSEC: start_time.milliseconds_since_epoch,
            "start_event_attr_key": "start_event_attr_val",
        }
        assert fake_start_span.start_time == to_nano(start_time.seconds_since_epoch)
        assert fake_start_span.end_time == to_nano(start_time.seconds_since_epoch)
        assert fake_start_span.context is not None
        assert fake_start_span.status.is_ok
        assert len(fake_start_span.events) == 0

        # Fake span for stop event
        fake_stop_span = exported_spans[1]
        assert fake_stop_span.name == "span_stop_event_for_test_span2"
        assert fake_stop_span.attributes == {
            StandardEventAttributeName.TIMESTAMP_MSEC: stop_time.milliseconds_since_epoch,
            "stop_event_attr_key": "stop_event_attr_val",
        }
        assert fake_stop_span.start_time == to_nano(stop_time.seconds_since_epoch)
        assert fake_stop_span.end_time == to_nano(stop_time.seconds_since_epoch)
        assert fake_stop_span.context is not None
        assert fake_stop_span.status.is_ok
        assert len(fake_stop_span.events) == 0

        # Real span for stop event
        exported_span = exported_spans[2]
        assert exported_span.name == "test_span2"
        assert exported_span.attributes == {
            "one_logger_span_id": test_span.id.int,
            "span_attrib_key": "span_attrib_val",
            StandardSpanAttributeName.DURATION_MSEC.value: 5310,
        }
        assert exported_span.context is not None
        assert exported_span.status.is_ok
        assert len(exported_span.events) == 2
        start_event = exported_span.events[0]
        assert start_event.name == "span_start"
        assert start_event.attributes == fake_start_span.attributes
        assert start_event.timestamp == to_nano(start_time.seconds_since_epoch)
        stop_event = exported_span.events[1]
        assert stop_event.name == "span_stop"
        assert stop_event.attributes == fake_stop_span.attributes
        assert stop_event.timestamp == to_nano(stop_time.seconds_since_epoch)

        # Make sure the relationship between the real span and the fake ones is set up correctly
        assert fake_start_span.parent is not None
        assert fake_start_span.parent.span_id == exported_span.context.span_id
        assert fake_stop_span.parent is not None
        assert fake_stop_span.parent.span_id == exported_span.context.span_id

    def test_export_error(self, exporter: OTelExporter, span_exporter: InMemorySpanExporter) -> None:
        """Test that error events are correctly exported."""
        # Create a test span
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=time.time(), perf_counter=100)
        test_span = Span.create(
            name="test_span1",
            start_time=start_time,
        )
        exporter.export_start(test_span)

        # Create an error event
        error_time = advance_time(start_time, 2.5)
        error_event = ErrorEvent(
            timestamp=error_time,
            error_message="Test error message",
            exception_type="RuntimeError",
            exception_message="test exception",
            exception_traceback="test exception traceback",
        )
        exporter.export_error(error_event, test_span)

        exporter._processor.force_flush()  # type: ignore[reportPrivateUsage]

        # Verify that the spans were exported correctly
        exported_spans = span_exporter.get_finished_spans()
        assert len(exported_spans) == 2  # fake spans for the start and error events

        assert exported_spans[0].name == "span_start_event_for_test_span1"
        # Verify the error event span
        error_span = exported_spans[1]
        assert error_span.name == "error_for_test_span1"
        assert error_span.attributes == {
            StandardEventAttributeName.ERROR_MESSAGE: "Test error message",
            StandardEventAttributeName.EXCEPTION_TYPE: "RuntimeError",
            StandardEventAttributeName.EXCEPTION_MESSAGE: "test exception",
            StandardEventAttributeName.EXCEPTION_TRACEBACK: "test exception traceback",
            StandardEventAttributeName.TIMESTAMP_MSEC: error_time.milliseconds_since_epoch,
        }
        assert error_span.start_time == to_nano(error_time.seconds_since_epoch)
        assert error_span.end_time == to_nano(error_time.seconds_since_epoch)
        assert error_span.context is not None
        assert not error_span.status.is_ok
        assert error_span.status.status_code == StatusCode.ERROR
        assert len(error_span.events) == 0

        # Add another error event
        next_error_event = ErrorEvent.create(error_message="Test error message 2")
        next_error_time = advance_time(error_time, 7)
        next_error_event._timestamp = next_error_time  # type: ignore[reportPrivateUsage]
        exporter.export_error(next_error_event, test_span)

        # Now, let's stop the span and confirm that both error events are exported with the span
        stop_time = advance_time(next_error_time, 1.0)
        test_span.stop(stop_time=stop_time)
        exporter.export_stop(test_span)
        exporter.close()

        exported_spans = span_exporter.get_finished_spans()
        # 4 fake spans for the start, stop and error events and one real span for the span
        assert len(exported_spans) == 5

        # Verify the error events are exported with the span
        assert exported_spans[0].name == "span_start_event_for_test_span1"
        assert exported_spans[1].name == "error_for_test_span1"
        assert exported_spans[2].name == "error_for_test_span1"
        assert exported_spans[3].name == "span_stop_event_for_test_span1"

        # Verify the error events are exported with the span
        real_span = exported_spans[4]
        assert real_span.attributes == {
            "one_logger_span_id": test_span.id.int,
            StandardSpanAttributeName.DURATION_MSEC.value: 10500,  # 2.5 seconds + 7 seconds + 1 second
        }
        assert len(real_span.events) == 4
        assert real_span.events[0].name == "span_start"
        assert real_span.events[1].name == "error"
        assert real_span.events[2].name == "error"
        assert real_span.events[3].name == "span_stop"
        assert real_span.status.status_code == StatusCode.ERROR

        # The attributes for the error events are reflected both in the fake spans for the errors and the
        # events for the real span
        assert exported_spans[1].attributes == real_span.events[1].attributes
        assert exported_spans[2].attributes == real_span.events[2].attributes

    @pytest.mark.parametrize("error_type", list(TelemetryDataError.ErrorType))
    def test_export_telemetry_data_error(self, exporter: OTelExporter, span_exporter: InMemorySpanExporter, error_type: TelemetryDataError.ErrorType) -> None:
        """Test that telemetry data errors are correctly exported."""
        timestamp = TracingTimestamp.for_timestamp(timestamp_sec=time.time(), perf_counter=100)

        telemetry_data_error = TelemetryDataError.create(
            error_type=error_type,
            error_message="Test telemetry data error message",
        )
        telemetry_data_error._timestamp = timestamp  # type: ignore[reportPrivateUsage]
        exporter.export_telemetry_data_error(telemetry_data_error)

        exporter._processor.force_flush()  # type: ignore[reportPrivateUsage]

        # Verify
        exported_spans = span_exporter.get_finished_spans()
        assert len(exported_spans) == 1  # fake span for telemetry data error
        telemetry_data_error_span = exported_spans[0]
        assert len(telemetry_data_error_span.events) == 1
        event = telemetry_data_error_span.events[0]
        assert telemetry_data_error_span.name == "span_for_telemetry_data_error"
        assert event.name == StandardEventName.TELEMETRY_DATA_ERROR.value
        assert event.attributes == {
            StandardEventAttributeName.TELEMETRY_DATA_ERROR_TYPE: error_type.value,
            StandardEventAttributeName.ERROR_MESSAGE: "Test telemetry data error message",
            StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch,
        }

    def test_export_event(self, exporter: OTelExporter, span_exporter: InMemorySpanExporter) -> None:
        """Test that events are correctly exported."""
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=time.time(), perf_counter=100)
        test_span = Span.create(
            name="test_span1",
            start_time=start_time,
        )
        exporter.export_start(test_span)

        # Create an event
        event1_timestamp = advance_time(start_time, 2.5)
        event1 = Event.create(name="event1", attributes=Attributes({"ev1_attr1": "ev1_val1"}), timestamp=event1_timestamp)
        exporter.export_event(event1, test_span)

        exporter._processor.force_flush()  # type: ignore[reportPrivateUsage]

        # Verify that the spans were exported correctly
        exported_spans = span_exporter.get_finished_spans()
        assert len(exported_spans) == 2  # fake spans for the start event and event1

        assert exported_spans[0].name == "span_start_event_for_test_span1"
        # Verify the event span
        event1_span = exported_spans[1]
        assert event1_span.name == "event1_event_for_test_span1"
        assert event1_span.attributes == {
            StandardEventAttributeName.TIMESTAMP_MSEC: event1_timestamp.milliseconds_since_epoch,
            "ev1_attr1": "ev1_val1",
        }
        assert event1_span.start_time == to_nano(event1_timestamp.seconds_since_epoch)
        assert event1_span.end_time == to_nano(event1_timestamp.seconds_since_epoch)
        assert event1_span.context is not None
        assert event1_span.status.is_ok
        assert len(event1_span.events) == 0

        # Now, let's stop the span and confirm that event1 is exported with the span
        stop_time = advance_time(event1_timestamp, 1.0)
        test_span.stop(stop_time=stop_time)
        exporter.export_stop(test_span)
        exporter.close()

        exported_spans = span_exporter.get_finished_spans()
        # 3 fake spans for the start, stop and event1 and one real span for the span
        assert len(exported_spans) == 4

        # Verify the events are exported with the span
        assert exported_spans[0].name == "span_start_event_for_test_span1"
        assert exported_spans[1].name == "event1_event_for_test_span1"
        assert exported_spans[2].name == "span_stop_event_for_test_span1"
        assert exported_spans[3].name == "test_span1"

        # Verify the events are exported with the span
        real_span = exported_spans[3]
        assert real_span.attributes == {
            "one_logger_span_id": test_span.id.int,
            StandardSpanAttributeName.DURATION_MSEC.value: 3500,  # 2.5 seconds + 1 second
        }
        assert len(real_span.events) == 3
        assert real_span.events[0].name == "span_start"
        assert real_span.events[1].name == "event1"
        assert real_span.events[2].name == "span_stop"
        assert real_span.status.is_ok

        # The attributes for the event are reflected both in the fake span for the event and the
        # events for the real span
        assert exported_spans[1].attributes == real_span.events[1].attributes

    def test_close(self, exporter: OTelExporter) -> None:
        """Test that the exporter is properly closed."""
        # Mock the provider's shutdown method
        exporter._provider.shutdown = MagicMock()  # type: ignore[reportPrivateUsage]

        # Close the exporter
        exporter.close()

        # Verify that the provider was shut down
        exporter._provider.shutdown.assert_called_once()  # type: ignore[reportPrivateUsage]
