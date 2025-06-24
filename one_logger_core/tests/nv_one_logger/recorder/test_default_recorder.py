# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DefaultRecorder class."""

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event, StandardEventAttributeName, StandardEventName, TelemetryDataError
from nv_one_logger.core.span import StandardSpanAttributeName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.recorder.default_recorder import (
    _DISABLE_EXPORTER_AFTER_N_FAILURES,  # type: ignore[reportPrivateUsage]
)
from nv_one_logger.recorder.default_recorder import ExportCustomizationMode  # type: ignore[reportPrivateUsage]
from nv_one_logger.recorder.default_recorder import _ExporterState  # type: ignore[reportPrivateUsage]
from nv_one_logger.recorder.default_recorder import DefaultRecorder


@pytest.fixture
def recorder(mock_exporter: Exporter) -> DefaultRecorder:
    """Fixture that sets up a DefaultRecorder."""
    return DefaultRecorder(exporters=[mock_exporter])


@pytest.fixture(autouse=True)
def configure_provider(recorder: DefaultRecorder, config: OneLoggerConfig) -> None:
    """Fixture that configures the TrainingTelemetryProvider."""
    # Reset the state of the singletons
    OneLoggerProvider.instance()._config = None
    OneLoggerProvider.instance()._recorder = None
    OneLoggerProvider.instance().configure(config, recorder)


class TestDefaultRecorder:
    """Tests for the DefaultRecorder class."""

    def test_start_span(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test starting a span with attributes."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(22000, 400, validate_timestamp=False)
            span = recorder.start(
                span_name="test_span",
                span_attributes=Attributes({"key": "value"}),
                start_event_attributes=Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 22000000, "start_key": "start_value"}),
            )

        # Verify span was created with correct attributes
        assert span.name_str == "test_span"
        assert span.attributes == Attributes({"key": "value"})
        assert span.start_event.timestamp == TracingTimestamp.for_timestamp(22000, 400, validate_timestamp=False)
        assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 22000000, "start_key": "start_value"})
        assert span.id == span.id

        # Verify exporter was called with correct span
        mock_exporter.export_start.assert_called_once()
        exported_span = mock_exporter.export_start.call_args[0][0]
        assert exported_span == span

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_start_span_custom_time(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test starting a span with attributes."""
        span = recorder.start(
            span_name="test_span",
            span_attributes=Attributes({"key": "value"}),
            start_event_attributes=Attributes({"start_key": "start_value"}),
            start_time=TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False),
        )

        # Verify span was created with correct attributes
        assert span.name_str == "test_span"
        assert span.attributes == Attributes({"key": "value"})
        assert span.start_event.timestamp == TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
        assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 24000000, "start_key": "start_value"})
        assert span.id == span.id

        # Verify exporter was called with correct span
        mock_exporter.export_start.assert_called_once()
        exported_span = mock_exporter.export_start.call_args[0][0]
        assert exported_span == span

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_stop_span(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test stopping a span with attributes."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(22000, 400, validate_timestamp=False)
            # First create a span
            span = recorder.start("test_span")

            mock_now.return_value = TracingTimestamp.for_timestamp(22070, 470, validate_timestamp=False)
            recorder.stop(
                span,
                stop_event_attributes=Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 22070000, "stop_key": "stop_value"}),
            )

        # Verify span was created with correct attributes
        assert span.name_str == "test_span"
        assert span.start_event.timestamp == TracingTimestamp.for_timestamp(22000, 400, validate_timestamp=False)
        assert span.stop_event
        assert span.stop_event.timestamp == TracingTimestamp.for_timestamp(22070, 470, validate_timestamp=False)
        assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 22070000, "stop_key": "stop_value"})
        assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 70000})

        # Verify exporter was called with correct span
        mock_exporter.export_stop.assert_called_once()
        exported_span = mock_exporter.export_stop.call_args[0][0]
        assert exported_span == span

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_stop_span_custom_time(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test stopping a span with attributes."""
        span = recorder.start("test_span", start_time=TracingTimestamp.for_timestamp(22000, 400, validate_timestamp=False))
        recorder.stop(
            span, stop_event_attributes=Attributes({"stop_key": "stop_value"}), stop_time=TracingTimestamp.for_timestamp(22150, 550, validate_timestamp=False)
        )

        # Verify span was created with correct attributes
        assert span.name_str == "test_span"
        assert span.start_event.timestamp == TracingTimestamp.for_timestamp(22000, 400, validate_timestamp=False)
        assert span.stop_event
        assert span.stop_event.timestamp == TracingTimestamp.for_timestamp(22150, 550, validate_timestamp=False)
        assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 22150000, "stop_key": "stop_value"})
        assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 150000})

        # Verify exporter was called with correct span
        mock_exporter.export_stop.assert_called_once()
        exported_span = mock_exporter.export_stop.call_args[0][0]
        assert exported_span == span

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_add_event(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test adding an event to a span."""
        # First create a span
        span = recorder.start("test_span")

        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24060, 160, validate_timestamp=False)
            event = Event.create("test_event", attributes=Attributes({"event_key": "event_value"}))
            recorder.event(span, event)

        # Verify event was added to span
        assert len(span.events) == 2
        assert span.events[0].name == StandardEventName.SPAN_START
        assert span.events[1] == event
        assert span.events[1].name == "test_event"
        assert span.events[1].attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 24060000, "event_key": "event_value"})

        # Verify exporter was called with correct event and span
        mock_exporter.export_event.assert_called_once()
        exported_event, exported_span = mock_exporter.export_event.call_args[0]
        assert exported_event == event
        assert exported_span == span

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_add_error(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test adding an error event to a span."""
        # First create a span
        span = recorder.start("test_span")

        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24070, 170, validate_timestamp=False)
            error = RuntimeError("test error")
            error_event = recorder.error(span, "test error message", error)

        # Verify error event was added to span
        assert len(span.events) == 2
        assert span.events[1] == error_event
        assert span.events[1].name == "error"
        assert span.events[1].error_message == "test error message"
        assert span.events[1].exception_type == "RuntimeError"
        assert span.events[1].timestamp == TracingTimestamp.for_timestamp(24070, 170, validate_timestamp=False)

        # Verify exporter was called with correct error event and span
        mock_exporter.export_error.assert_called_once()
        exported_event, exported_span = mock_exporter.export_error.call_args[0]
        assert exported_event == error_event
        assert exported_span == span

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_multiple_exporters(self) -> None:
        """Test that multiple exporters receive the same data."""
        mock_exporter1 = MagicMock(spec=Exporter)
        mock_exporter2 = MagicMock(spec=Exporter)
        recorder = DefaultRecorder(exporters=[mock_exporter1, mock_exporter2])

        # Create and stop a span
        span = recorder.start("test_span")
        recorder.stop(span)

        # Verify both exporters received the same data
        mock_exporter1.export_start.assert_called_once()
        mock_exporter2.export_start.assert_called_once()
        mock_exporter1.export_stop.assert_called_once()
        mock_exporter2.export_stop.assert_called_once()

        # Verify the spans passed to both exporters are the same
        span1 = mock_exporter1.export_start.call_args[0][0]
        span2 = mock_exporter2.export_start.call_args[0][0]
        assert span1.id == span2.id

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter1.export_telemetry_data_error.assert_not_called()
        mock_exporter2.export_telemetry_data_error.assert_not_called()

    def test_close(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test closing the recorder with an active span."""
        # Create a span but don't stop it
        span = recorder.start("test_span")

        # Close the recorder
        recorder.close()

        # Verify the span was stopped and exported
        assert span.stop_event is not None
        mock_exporter.export_stop.assert_called_once()
        mock_exporter.close.assert_called_once()

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_whitelisted_spans(self) -> None:
        """Test that only whitelisted spans are exported."""
        mock_exporter1 = MagicMock(spec=Exporter)
        mock_exporter2 = MagicMock(spec=Exporter)
        recorder = DefaultRecorder(
            exporters=[mock_exporter1, mock_exporter2],
            export_customization_mode=ExportCustomizationMode.WHITELIST_SPANS,
            span_name_filter=["whitelisted_span", "another_whitelisted_span"],
        )

        span1 = recorder.start("span1")
        whitelisted_span = recorder.start("whitelisted_span")
        recorder.stop(whitelisted_span)

        span2 = recorder.start("span2")
        another_whitelisted_span = recorder.start("another_whitelisted_span")
        recorder.stop(another_whitelisted_span)
        recorder.stop(span2)
        recorder.stop(span1)

        for exporter in [mock_exporter1, mock_exporter2]:
            assert exporter.export_start.call_count == 2
            assert exporter.export_start.call_args_list[0].args[0].name == "whitelisted_span"
            assert exporter.export_start.call_args_list[1].args[0].name == "another_whitelisted_span"

            assert exporter.export_stop.call_count == 2
            assert exporter.export_stop.call_args_list[0].args[0].name == "whitelisted_span"
            assert exporter.export_stop.call_args_list[1].args[0].name == "another_whitelisted_span"

            # No telemetry data error was reported as there was no error during the export call.
            exporter.export_telemetry_data_error.assert_not_called()

    def test_blacklisted_spans(self) -> None:
        """Test that only whitelisted spans are exported."""
        mock_exporter1 = MagicMock(spec=Exporter)
        mock_exporter2 = MagicMock(spec=Exporter)
        recorder = DefaultRecorder(
            exporters=[mock_exporter1, mock_exporter2],
            export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS,
            span_name_filter=["blacklisted_spans", "another_blacklisted_spans"],
        )

        span1 = recorder.start("span1")
        whitelisted_span = recorder.start("blacklisted_spans")
        recorder.stop(whitelisted_span)

        span2 = recorder.start("span2")
        another_whitelisted_span = recorder.start("another_blacklisted_spans")
        recorder.stop(another_whitelisted_span)
        recorder.stop(span2)
        recorder.stop(span1)

        for exporter in [mock_exporter1, mock_exporter2]:
            assert exporter.export_start.call_count == 2
            assert exporter.export_start.call_args_list[0].args[0].name == "span1"
            assert exporter.export_start.call_args_list[1].args[0].name == "span2"

            assert exporter.export_stop.call_count == 2
            assert exporter.export_stop.call_args_list[0].args[0].name == "span2"
            assert exporter.export_stop.call_args_list[1].args[0].name == "span1"

            # No telemetry data error was reported as there was no error during the export call.
            exporter.export_telemetry_data_error.assert_not_called()

    def test_telemetry_data_error(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test recording a telemetry data issue."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24080, 180, validate_timestamp=False)
            recorder.telemetry_data_error("Failed to collect telemetry data", attributes=Attributes({"error_code": 500}))

        mock_exporter.export_telemetry_data_error.assert_called_once()
        exported_error = mock_exporter.export_telemetry_data_error.call_args[0][0]
        assert exported_error.error_type == TelemetryDataError.ErrorType.NO_TELEMETRY_DATA
        assert exported_error.error_message == "Failed to collect telemetry data"
        assert exported_error.timestamp == TracingTimestamp.for_timestamp(24080, 180, validate_timestamp=False)
        assert exported_error.attributes["error_code"].value == 500

        # No more exports to the exporter.
        mock_exporter.reset_mock()
        recorder.telemetry_data_error("another error", attributes=Attributes({"error_code": 501}))
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_span_hierarchy(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test that we set the parent-child relationship between spans correctly."""
        # span1->span2->span3
        #             ->span4
        #             ->span5
        #      ->span6
        # span7
        span1 = recorder.start("test_span1")
        span2 = recorder.start("test_span2")
        span3 = recorder.start("test_span3")
        recorder.stop(span3)

        span4 = recorder.start("test_span4")
        recorder.stop(span4)

        span5 = recorder.start("test_span5")
        recorder.event(span5, Event.create("test_event"))
        recorder.stop(span5)

        recorder.stop(span2)
        span6 = recorder.start("test_span6")
        recorder.stop(span6)
        recorder.stop(span1)

        span7 = recorder.start("test_span7")
        recorder.stop(span7)

        assert span1.parent_span is None
        assert span2.parent_span is span1
        assert span3.parent_span is span2
        assert span4.parent_span is span2
        assert span5.parent_span is span2
        assert span6.parent_span is span1
        assert span7.parent_span is None

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_span_hierarchy_malformed(self, recorder: DefaultRecorder, mock_exporter: MagicMock) -> None:
        """Test that even if the caller uses the library incorrectly by stopping a parent span before its children, we avoid a crash.

        In this case, we should use the closest active ancestor.
        """
        # span1->span2->span3
        #             ->span4
        #             ->span5
        #      ->span6
        # span7
        span1 = recorder.start("test_span1")
        span2 = recorder.start("test_span2")
        span3 = recorder.start("test_span3")

        # This is wrong, but we should handle it gracefully.
        recorder.stop(span2)

        recorder.stop(span3)

        span4 = recorder.start("test_span4")
        assert span1.parent_span is None
        assert span2.parent_span is span1
        assert span3.parent_span is span2
        assert span4.parent_span is span1

        # No telemetry data error was reported as there was no error during the export call.
        mock_exporter.export_telemetry_data_error.assert_not_called()

    def test_exporter_fails_from_start(self) -> None:
        """Test behavior when an exporter fails immediately from the start."""
        # Create two exporters - one that fails and one that works
        failing_exporter = MagicMock(spec=Exporter)
        working_exporter = MagicMock(spec=Exporter)

        # Configure the failing exporter to raise an exception on all export calls
        failing_exporter.export_start.side_effect = Exception("Export failed")
        failing_exporter.export_stop.side_effect = Exception("Export failed")
        failing_exporter.export_event.side_effect = Exception("Export failed")
        failing_exporter.export_error.side_effect = Exception("Export failed")

        recorder = DefaultRecorder(exporters=[failing_exporter, working_exporter])

        # Make enough calls to the failing exporter to cause it to get disabled. We call 3 functions that each call the exporter.
        num_calls = math.ceil(_DISABLE_EXPORTER_AFTER_N_FAILURES / 3) + 1  # type: ignore[reportPrivateUsage]
        for _ in range(num_calls):
            span = recorder.start("test_span")
            recorder.event(span, Event.create("test_event"))
            recorder.stop(span)

        # Verify the working exporter received all calls
        assert working_exporter.export_start.call_count == num_calls
        assert working_exporter.export_event.call_count == num_calls
        assert working_exporter.export_stop.call_count == num_calls
        # No error occured during the export calls.
        working_exporter.export_telemetry_data_error.assert_not_called()

        # Verify the failing exporter got disabled and was not called after the threshold was reached.
        assert failing_exporter.export_start.call_count < num_calls
        assert failing_exporter.export_event.call_count < num_calls
        assert failing_exporter.export_stop.call_count < num_calls
        # The first export of span start fails, so we call export_telemetry_data_error, which succeeds.
        # Then we have _DISABLE_EXPORTER_AFTER_N_FAILURES consecutive failures before we disable the exporter.
        assert (
            failing_exporter.export_start.call_count + failing_exporter.export_event.call_count + failing_exporter.export_stop.call_count
            == _DISABLE_EXPORTER_AFTER_N_FAILURES
        )
        assert failing_exporter.export_telemetry_data_error.call_count == 1

        assert recorder._exporter_state[working_exporter] == _ExporterState(  # type: ignore[reportPrivateUsage]
            num_errors=0,
            force_disabled=False,
            any_data_exported_successfully=True,
            telemetry_data_error_reported=False,
        )
        assert recorder._exporter_state[failing_exporter] == _ExporterState(  # type: ignore[reportPrivateUsage]
            num_errors=_DISABLE_EXPORTER_AFTER_N_FAILURES,
            force_disabled=True,
            any_data_exported_successfully=False,
            telemetry_data_error_reported=True,
        )

    def test_exporter_fails_after_success(self) -> None:
        """Test behavior when an exporter works initially but fails later."""
        # Create two exporters - one that will fail later and one that works consistently
        failing_exporter = MagicMock(spec=Exporter)
        working_exporter = MagicMock(spec=Exporter)

        failing_exporter.export_event.side_effect = Exception("Export failed")

        recorder = DefaultRecorder(exporters=[failing_exporter, working_exporter])

        # Create and stop first span (should work for both exporters)
        span1 = recorder.start("first_span")
        recorder.stop(span1)

        span2 = recorder.start("first_span")
        # Send a bunch of events
        for _ in range(_DISABLE_EXPORTER_AFTER_N_FAILURES * 2):
            recorder.event(span2, Event.create("test_event"))

        assert working_exporter.export_start.call_count == 2
        assert working_exporter.export_stop.call_count == 1
        assert working_exporter.export_event.call_count == _DISABLE_EXPORTER_AFTER_N_FAILURES * 2
        working_exporter.export_telemetry_data_error.assert_not_called()

        assert failing_exporter.export_start.call_count == 2
        assert failing_exporter.export_stop.call_count == 1
        assert failing_exporter.export_event.call_count == _DISABLE_EXPORTER_AFTER_N_FAILURES
        assert failing_exporter.export_telemetry_data_error.call_count == 1

    def test_intermittent_failures(self) -> None:
        """Test behavior when an exporter fails intermittently."""
        # Create two exporters - one that will fail later and one that works consistently
        failing_exporter = MagicMock(spec=Exporter)
        working_exporter = MagicMock(spec=Exporter)

        def failing_export_stop(*args: Any, **kwargs: Any) -> None:
            if failing_exporter.export_stop.call_count % 3 == 0:
                raise Exception("Export failed")

        failing_exporter.export_stop.side_effect = failing_export_stop

        recorder = DefaultRecorder(exporters=[failing_exporter, working_exporter])

        num_spans = _DISABLE_EXPORTER_AFTER_N_FAILURES * 3 + 1
        for _ in range(num_spans):
            span = recorder.start("test_span")
            recorder.event(span, Event.create("test_event"))
            recorder.stop(span)

        assert working_exporter.export_start.call_count == num_spans
        assert working_exporter.export_stop.call_count == num_spans
        assert working_exporter.export_event.call_count == num_spans
        working_exporter.export_telemetry_data_error.assert_not_called()

        assert failing_exporter.export_start.call_count == _DISABLE_EXPORTER_AFTER_N_FAILURES * 3
        assert failing_exporter.export_stop.call_count == _DISABLE_EXPORTER_AFTER_N_FAILURES * 3
        assert failing_exporter.export_event.call_count == _DISABLE_EXPORTER_AFTER_N_FAILURES * 3
        assert failing_exporter.export_telemetry_data_error.call_count == 1
