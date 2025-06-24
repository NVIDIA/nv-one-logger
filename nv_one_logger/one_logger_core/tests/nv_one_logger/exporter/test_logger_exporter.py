# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LoggerExporter class."""

import logging
import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import ErrorEvent, Event, StandardEventAttributeName, TelemetryDataError
from nv_one_logger.core.span import Span
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.logger_exporter import LoggerExporter

_MOCK_PID = 123


class TestLoggerExporter:
    """Tests for the LoggerExporter class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self) -> Generator[None, None, None]:
        """Set up and tear down the test environment."""
        self.patcher = patch("os.getpid")
        self.mock_function = self.patcher.start()
        self.mock_function.return_value = _MOCK_PID

        self.logger = MagicMock(spec=logging.Logger)
        self.exporter = LoggerExporter(self.logger)
        assert not self.exporter.ready
        self.exporter.initialize()
        assert self.exporter.ready
        self.pid = os.getpid()
        self.maxDiff = None

        yield

        self.exporter.close()
        assert self.exporter.closed
        self.patcher.stop()

    def test_export_start_with_no_previous_span(self) -> None:
        """Test export_start with no previous span doesn't produce any log entries yet."""
        span = Span.create("test_span")
        self.exporter.export_start(span)
        self.logger.info.assert_not_called()

    def test_export_start_with_previous_span(self) -> None:
        """Test export_start with a previous span."""
        # Set up previous span
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(23456.0, 100.4, validate_timestamp=False)
            prev_span = Span.create("test_span1")
            self.exporter.export_start(prev_span)

        # New span
        new_span = Span.create("test_span2")
        self.exporter.export_start(new_span)

        # Verify previous span was recorded
        self.logger.info.assert_called_once()
        actual_log_entry = self.logger.info.call_args[0][0]

        expected_log_entry = (
            '[type="start" | '
            + 'start_event.name="span_start" | start_event.attributes.timestamp_msec=23456000 | '
            + f'pid={_MOCK_PID} | name="test_span1" | id="{prev_span.id}" | count=1]'
        )

        assert actual_log_entry == expected_log_entry

        expected_deserialized_log = {
            "type": "start",
            "pid": _MOCK_PID,
            "name": "test_span1",
            "id": f"{prev_span.id}",
            "count": 1,
            "start_event": {
                "name": "span_start",
                "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 23456000},
            },
        }
        assert LoggerExporter.deserialize_log(actual_log_entry) == expected_deserialized_log

        # Now let's make sure _format_log and deserialize_log are inverse of each other
        assert LoggerExporter.deserialize_log(self.exporter._format_log(expected_deserialized_log)) == expected_deserialized_log

    def test_export_event(self) -> None:
        """Test export_event functionality."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span = Span.create("test_span")
            self.exporter.export_start(span)

            mock_now.return_value = TracingTimestamp.for_timestamp(24060, 160, validate_timestamp=False)
            event = span.add_event(Event.create("test_event"))
            self.exporter.export_event(event, span)

        assert self.logger.info.call_count == 2
        actual_log_entries = [call.args[0] for call in self.logger.info.call_args_list]
        expected_log_entries = [
            # start span
            '[type="start" | '
            + 'start_event.name="span_start" | start_event.attributes.timestamp_msec=24000000 | '
            + f'pid={_MOCK_PID} | name="test_span" | id="{span.id}" | count=1]',
            # event
            f'[type="event" | span_id="{span.id}" | pid={_MOCK_PID} ' + '| event.name="test_event" ' + "| event.attributes.timestamp_msec=24060000 | count=2]",
        ]
        assert actual_log_entries == expected_log_entries

        expected_deserialized_event = {
            "type": "event",
            "span_id": f"{span.id}",
            "pid": _MOCK_PID,
            "event": {
                "name": "test_event",
                "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24060000},
            },
            "count": 2,
        }
        assert LoggerExporter.deserialize_log(actual_log_entries[1]) == expected_deserialized_event

        # Now let's make sure _format_log and deserialize_log are inverse of each other
        assert LoggerExporter.deserialize_log(self.exporter._format_log(expected_deserialized_event)) == expected_deserialized_event

    def test_export_multiple_spans_with_attributes(self) -> None:
        """Test a case where we have multiple spans and events with attributes."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span1 = Span.create("test_span1", span_attributes=Attributes({"sp1_att_k1": "sp1_att_v1"}))
            self.exporter.export_start(span1)

            mock_now.return_value = TracingTimestamp.for_timestamp(24010, 110, validate_timestamp=False)
            span2 = Span.create(
                "test_span2",
                span_attributes=Attributes({"sp2_att_k1": False, "sp2_att_k2": 55.45}),
                start_event_attributes=Attributes({"sp2_ev1_att_k1": "sp2_ev1_att_v1", StandardEventAttributeName.TIMESTAMP_MSEC: 24010000}),
            )

            mock_now.return_value = TracingTimestamp.for_timestamp(24020, 120, validate_timestamp=False)

            self.exporter.export_start(span2)

            mock_now.return_value = TracingTimestamp.for_timestamp(24050, 150, validate_timestamp=False)
            span2.stop(stop_event_attributes=Attributes({"sp2_se_k1": ["sp2_se_v1", "sp2_se_v2"]}))
            self.exporter.export_stop(span2)

            mock_now.return_value = TracingTimestamp.for_timestamp(24100, 200, validate_timestamp=False)
            event1 = span1.add_event(
                Event.create(
                    "test_event1",
                    attributes=Attributes({"sp1_ev1_att_k1": 44, "sp1_ev1_att_k2": [1, 2, 3], StandardEventAttributeName.TIMESTAMP_MSEC: 24100000}),
                )
            )
            self.exporter.export_event(event1, span1)

            mock_now.return_value = TracingTimestamp.for_timestamp(24150, 250, validate_timestamp=False)
            span1.stop(
                stop_event_attributes=Attributes({"sp1_se_att_k1": 44, "sp1_se_att_k2": ["el1", "el2"], StandardEventAttributeName.TIMESTAMP_MSEC: 24150000})
            )
            self.exporter.export_stop(span1)

            self.exporter.close()

        actual_log_entries = [call.args[0] for call in self.logger.info.call_args_list]
        assert len(actual_log_entries) == 4
        expected_log_entries = [
            # 0. span1 starting (start record)
            '[type="start" | start_event.name="span_start" '
            + "| start_event.attributes.timestamp_msec=24000000 "
            + f'| pid=123 | name="test_span1" | id="{span1.id}" | count=1 | attributes.sp1_att_k1="sp1_att_v1"]',
            # 1. span2 starting and stopping (complete record)
            '[type="complete" '
            + '| stop_event.name="span_stop" '
            + "| stop_event.attributes.timestamp_msec=24050000 "
            + '| stop_event.attributes.sp2_se_k1=["sp2_se_v1", "sp2_se_v2"] '
            + '| start_event.name="span_start" '
            + "| start_event.attributes.timestamp_msec=24010000 "
            + '| start_event.attributes.sp2_ev1_att_k1="sp2_ev1_att_v1" '
            + f'| pid=123 | name="test_span2" | id="{span2.id}" | count=2 '
            + "| attributes.sp2_att_k2=55.45 | attributes.sp2_att_k1=false "
            + "| attributes.duration_msec=40000]",
            # 2. event for span1 (event record)
            f'[type="event" | span_id="{span1.id}" | pid=123 '
            + '| event.name="test_event1" '
            + "| event.attributes.timestamp_msec=24100000 "
            + "| event.attributes.sp1_ev1_att_k2=[1, 2, 3] | event.attributes.sp1_ev1_att_k1=44 | count=3]",
            # 3. span1 stopping (stop record)
            '[type="stop" '
            + '| stop_event.name="span_stop" '
            + "| stop_event.attributes.timestamp_msec=24150000 "
            + '| stop_event.attributes.sp1_se_att_k2=["el1", "el2"] | stop_event.attributes.sp1_se_att_k1=44 '
            + f'| pid={_MOCK_PID} | name="test_span1" | id="{span1.id}" | count=4 '
            + '| attributes.sp1_att_k1="sp1_att_v1" '
            + "| attributes.duration_msec=150000]",
        ]
        assert actual_log_entries == expected_log_entries

        # Now, let's test the deserialization of the most complex log entry
        expected_deserialized_event = {
            "type": "complete",
            "stop_event": {
                "name": "span_stop",
                "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24050000, "sp2_se_k1": ["sp2_se_v1", "sp2_se_v2"]},
            },
            "start_event": {
                "name": "span_start",
                "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24010000, "sp2_ev1_att_k1": "sp2_ev1_att_v1"},
            },
            "pid": 123,
            "name": "test_span2",
            "id": f"{span2.id}",
            "count": 2,
            "attributes": {"sp2_att_k2": 55.45, "sp2_att_k1": False, "duration_msec": 40000},
        }

        assert LoggerExporter.deserialize_log(actual_log_entries[1]) == expected_deserialized_event

        # Now let's make sure _format_log and deserialize_log are inverse of each other
        assert LoggerExporter.deserialize_log(self.exporter._format_log(expected_deserialized_event)) == expected_deserialized_event

    def test_export_error(self) -> None:
        """Test export_error functionality."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span = Span.create("test_span")
            self.exporter.export_start(span)

        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24060, 160, validate_timestamp=False)
            ex = RuntimeError("error_exception")
            error_event = ErrorEvent.create(error_message="error_msg", exception=ex)
            self.exporter.export_error(error_event, span)

        assert self.logger.info.call_count == 2
        actual_log_entries = [call.args[0] for call in self.logger.info.call_args_list]
        expected_log_entries = [
            # start span
            '[type="start" '
            + '| start_event.name="span_start" '
            + "| start_event.attributes.timestamp_msec=24000000 "
            + f'| pid={_MOCK_PID} | name="test_span" | id="{span.id}" | count=1]',
            # error event
            f'[type="application_error" | span_id="{span.id}" | pid={_MOCK_PID} '
            + '| event.name="error" '
            + "| event.attributes.timestamp_msec=24060000 "
            + '| event.attributes.exception_type="RuntimeError" | event.attributes.exception_message="error_exception" '
            + '| event.attributes.error_message="error_msg" | count=2]',
        ]
        assert actual_log_entries == expected_log_entries

        expected_deserialized_error_event = {
            "type": "application_error",
            "span_id": f"{span.id}",
            "pid": _MOCK_PID,
            "event": {
                "name": "error",
                "attributes": {
                    StandardEventAttributeName.EXCEPTION_TYPE: "RuntimeError",
                    StandardEventAttributeName.EXCEPTION_MESSAGE: "error_exception",
                    StandardEventAttributeName.ERROR_MESSAGE: "error_msg",
                    StandardEventAttributeName.TIMESTAMP_MSEC: 24060000,
                },
            },
            "count": 2,
        }

        assert LoggerExporter.deserialize_log(actual_log_entries[1]) == expected_deserialized_error_event

        # Now let's make sure _format_log and deserialize_log are inverse of each other
        assert LoggerExporter.deserialize_log(self.exporter._format_log(expected_deserialized_error_event)) == expected_deserialized_error_event

    def test_export_telemetry_data_error(self) -> None:
        """Test export_telemetry_data_error functionality."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            error = TelemetryDataError.create(error_type=TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA, error_message="error_msg")
            self.exporter.export_telemetry_data_error(error)

        assert self.logger.info.call_count == 1
        actual_log_entries = [call.args[0] for call in self.logger.info.call_args_list]
        expected_log_entries = [
            f'[type="telemetry_data_error" | pid={_MOCK_PID} | error.name="telemetry_data_error" '
            + "| error.attributes.timestamp_msec=24000000 "
            + '| error.attributes.telemetry_data_error_type="incomplete_telemetry_data" | error.attributes.error_message="error_msg" | count=1]',
        ]
        assert actual_log_entries == expected_log_entries

    def test_close_with_unclosed_span(self) -> None:
        """Test close method with an unclosed span."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span = Span.create("test_span")

        self.exporter.export_start(span)
        self.exporter.close()
        expected_log_entry = (
            '[type="start" '
            + '| start_event.name="span_start" '
            + f"| start_event.attributes.timestamp_msec=24000000 | pid={_MOCK_PID} "
            + f'| name="test_span" | id="{span.id}" | count=1]'
        )
        actual_log_entry = self.logger.info.call_args_list[0].args[0]
        assert actual_log_entry == expected_log_entry
