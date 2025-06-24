# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the FileExporter class."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import ErrorEvent, Event, StandardEventAttributeName, TelemetryDataError
from nv_one_logger.core.span import Span
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import ExportError
from nv_one_logger.exporter.file_exporter import FileExporter

_MOCK_PID = 123


class TestFileExporter:
    """Tests for the FileExporter class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self) -> None:
        """Set up and tear down the test environment."""
        self.patcher = patch("os.getpid")
        self.mock_function = self.patcher.start()
        self.mock_function.return_value = _MOCK_PID

        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.exporter = FileExporter(Path(self.temp_file.name))
        assert not self.exporter.ready
        assert not self.exporter.closed
        self.exporter.initialize()
        assert self.exporter.ready
        self.pid = os.getpid()
        self.maxDiff = None

        yield

        self.exporter.close()
        assert self.exporter.closed
        os.unlink(self.temp_file.name)
        self.patcher.stop()

    def _extract_file_payload(self) -> List[Dict[str, Any]]:
        """Extract and parse the file payload from the temporary file.

        Returns:
            List[Dict[str, Any]]: List of parsed JSON objects from the file.
            Empty lines are excluded from the result.
        """
        # Accessing the protected member is needed here because
        # the file has to be closed so that the I/O is completed.
        # Calling exporter.close() is not a good option because this
        # function sometimes writes extra contents to the file  (e.g.,
        # writing the last started but not stopped span to the file).
        if self.exporter._file:
            self.exporter._file.close()
        with open(self.temp_file.name, mode="rb") as f:
            f.seek(0)
            file_payload = f.read().decode().strip().split("\n")
            # Filter out empty strings
            file_payload = [fp for fp in file_payload if fp]
        return [json.loads(fp) for fp in file_payload]

    def test_export_start_with_no_previous_span(self) -> None:
        """Test export_start with no previous span doesn't produce any log entries yet."""
        span = Span.create("test_span")
        self.exporter.export_start(span)

        assert self._extract_file_payload() == []

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
        file_payload = self._extract_file_payload()
        assert len(file_payload) == 1

        expected_file_entry = {
            "type": "start",
            "pid": _MOCK_PID,
            "id": str(prev_span.id),
            "name": "test_span1",
            "start_event": {
                "name": "span_start",
                "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 23456000},
            },
            "count": 1,
        }
        assert file_payload[0] == expected_file_entry

    def test_export_event(self) -> None:
        """Test export_event functionality."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span = Span.create("test_span")
            self.exporter.export_start(span)

            mock_now.return_value = TracingTimestamp.for_timestamp(24060, 160, validate_timestamp=False)
            event = span.add_event(Event.create("test_event"))
            self.exporter.export_event(event, span)

            file_payload = self._extract_file_payload()
            assert len(file_payload) == 2

        expected_file_entries = [
            {
                "type": "start",
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24000000},
                },
                "pid": _MOCK_PID,
                "name": "test_span",
                "id": str(span.id),
                "count": 1,
            },
            {
                "type": "event",
                "span_id": str(span.id),
                "pid": _MOCK_PID,
                "event": {
                    "name": "test_event",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24060000},
                },
                "count": 2,
            },
        ]

        assert file_payload == expected_file_entries

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
                start_event_attributes=Attributes({"sp2_ev1_att_k1": "sp2_ev1_att_v1"}),
            )

            mock_now.return_value = TracingTimestamp.for_timestamp(24020, 120, validate_timestamp=False)
            self.exporter.export_start(span2)

            mock_now.return_value = TracingTimestamp.for_timestamp(24050, 150, validate_timestamp=False)
            span2.stop(stop_event_attributes=Attributes({"sp2_se_k1": ["sp2_se_v1", "sp2_se_v2"]}))
            self.exporter.export_stop(span2)

            mock_now.return_value = TracingTimestamp.for_timestamp(24100, 200, validate_timestamp=False)
            event1 = span1.add_event(Event.create("test_event1", attributes=Attributes({"sp1_ev1_att_k1": 44, "sp1_ev1_att_k2": [1, 2, 3]})))
            self.exporter.export_event(event1, span1)

            mock_now.return_value = TracingTimestamp.for_timestamp(24150, 250, validate_timestamp=False)
            span1.stop(stop_event_attributes=Attributes({"sp1_se_att_k1": 44, "sp1_se_att_k2": ["el1", "el2"]}))
            self.exporter.export_stop(span1)

            self.exporter.close()

            file_payload = self._extract_file_payload()
            assert len(file_payload) == 4

        expected_file_entries = [
            # 0. span1 starting (start record)
            {
                "type": "start",
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24000000},
                },
                "pid": _MOCK_PID,
                "name": "test_span1",
                "id": str(span1.id),
                "count": 1,
                "attributes": {"sp1_att_k1": "sp1_att_v1"},
            },
            # 1. span2 starting and stopping (complete record)
            {
                "type": "complete",
                "stop_event": {
                    "name": "span_stop",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24050000, "sp2_se_k1": ["sp2_se_v1", "sp2_se_v2"]},
                },
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24010000, "sp2_ev1_att_k1": "sp2_ev1_att_v1"},
                },
                "pid": _MOCK_PID,
                "name": "test_span2",
                "id": str(span2.id),
                "count": 2,
                "attributes": {"sp2_att_k2": 55.45, "sp2_att_k1": False, "duration_msec": 40000},
            },
            # 2. event for span1 (event record)
            {
                "type": "event",
                "span_id": str(span1.id),
                "pid": _MOCK_PID,
                "event": {
                    "name": "test_event1",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24100000, "sp1_ev1_att_k2": [1, 2, 3], "sp1_ev1_att_k1": 44},
                },
                "count": 3,
            },
            # 3. span1 stopping (stop record)
            {
                "type": "stop",
                "stop_event": {
                    "name": "span_stop",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24150000, "sp1_se_att_k2": ["el1", "el2"], "sp1_se_att_k1": 44},
                },
                "pid": _MOCK_PID,
                "name": "test_span1",
                "id": str(span1.id),
                "count": 4,
                "attributes": {"sp1_att_k1": "sp1_att_v1", "duration_msec": 150000},
            },
        ]

        assert file_payload == expected_file_entries

    def test_export_overlapping_spans(self) -> None:
        """Test a case where we have multiple overlapping spans."""
        # set up a span1.start -> span2.start -> span1.stop -> span3.start -> span2.stop -> span3.stop scenario
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span1 = Span.create("test_span1")
            self.exporter.export_start(span1)

            mock_now.return_value = TracingTimestamp.for_timestamp(24010, 110, validate_timestamp=False)
            span2 = Span.create("test_span2")
            self.exporter.export_start(span2)

            mock_now.return_value = TracingTimestamp.for_timestamp(24050, 150, validate_timestamp=False)
            span1.stop()
            self.exporter.export_stop(span1)

            mock_now.return_value = TracingTimestamp.for_timestamp(24080, 180, validate_timestamp=False)
            span3 = Span.create("test_span3")
            self.exporter.export_start(span3)

            mock_now.return_value = TracingTimestamp.for_timestamp(24100, 200, validate_timestamp=False)
            span2.stop()
            self.exporter.export_stop(span2)

            mock_now.return_value = TracingTimestamp.for_timestamp(24130, 230, validate_timestamp=False)
            span3.stop()
            self.exporter.export_stop(span3)

            self.exporter.close()

            file_payload = self._extract_file_payload()
            assert len(file_payload) == 6

        expected_file_entries = [
            # 0. span1 starting (start record)
            {
                "type": "start",
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24000000},
                },
                "pid": _MOCK_PID,
                "name": "test_span1",
                "id": str(span1.id),
                "count": 1,
            },
            # 1. span2 starting  (start record)
            {
                "type": "start",
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24010000},
                },
                "pid": _MOCK_PID,
                "name": "test_span2",
                "id": str(span2.id),
                "count": 2,
            },
            # 2. span1 stopping (stop record)
            {
                "type": "stop",
                "stop_event": {
                    "name": "span_stop",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24050000},
                },
                "pid": _MOCK_PID,
                "name": "test_span1",
                "id": str(span1.id),
                "count": 3,
                "attributes": {"duration_msec": 50000},
            },
            # 3. span3 starting (start record)
            {
                "type": "start",
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24080000},
                },
                "pid": _MOCK_PID,
                "name": "test_span3",
                "id": str(span3.id),
                "count": 4,
            },
            # 4. span2 stopping (stop record)
            {
                "type": "stop",
                "stop_event": {
                    "name": "span_stop",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24100000},
                },
                "pid": _MOCK_PID,
                "name": "test_span2",
                "id": str(span2.id),
                "count": 5,
                "attributes": {"duration_msec": 90000},
            },
            # 5. span3 stopping (stop record)
            {
                "type": "stop",
                "stop_event": {
                    "name": "span_stop",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24130000},
                },
                "pid": _MOCK_PID,
                "name": "test_span3",
                "id": str(span3.id),
                "count": 6,
                "attributes": {"duration_msec": 50000},
            },
        ]

        assert file_payload == expected_file_entries

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

            file_payload = self._extract_file_payload()
            assert len(file_payload) == 2

        expected_file_entries = [
            {
                "type": "start",
                "start_event": {
                    "name": "span_start",
                    "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24000000},
                },
                "pid": _MOCK_PID,
                "name": "test_span",
                "id": str(span.id),
                "count": 1,
            },
            {
                "type": "application_error",
                "span_id": str(span.id),
                "pid": _MOCK_PID,
                "event": {
                    "name": "error",
                    "attributes": {
                        "exception_type": "RuntimeError",
                        "exception_message": "error_exception",
                        "error_message": "error_msg",
                        StandardEventAttributeName.TIMESTAMP_MSEC: 24060000,
                    },
                },
                "count": 2,
            },
        ]
        assert file_payload == expected_file_entries

    def test_export_telemetry_data_error(self) -> None:
        """Test export_telemetry_data_error functionality."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(1234567890, 100.4, validate_timestamp=False)

            error_event = TelemetryDataError.create(
                error_type=TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA,
                error_message="telemetry_error_msg",
            )
            self.exporter.export_telemetry_data_error(error_event)

            file_payload = self._extract_file_payload()
            assert len(file_payload) == 1

            expected_file_entries = [
                {
                    "type": "telemetry_data_error",
                    "error": {
                        "name": "telemetry_data_error",
                        "attributes": {
                            "timestamp_msec": 1234567890000,
                            "error_message": "telemetry_error_msg",
                            "telemetry_data_error_type": "incomplete_telemetry_data",
                        },
                    },
                    "pid": _MOCK_PID,
                    "count": 1,
                },
            ]
            assert file_payload == expected_file_entries

    def test_close_with_unclosed_span(self) -> None:
        """Test close method with an unclosed span."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            mock_now.return_value = TracingTimestamp.for_timestamp(24000, 100, validate_timestamp=False)
            span = Span.create("test_span")

        self.exporter.export_start(span)
        self.exporter.close()

        file_payload = self._extract_file_payload()
        assert len(file_payload) == 1

        expected_file_entry = {
            "type": "start",
            "start_event": {
                "name": "span_start",
                "attributes": {StandardEventAttributeName.TIMESTAMP_MSEC: 24000000},
            },
            "pid": _MOCK_PID,
            "name": "test_span",
            "id": str(span.id),
            "count": 1,
        }
        assert file_payload[0] == expected_file_entry


class TestFileExporterStateErrors:
    """Tests for the FileExporter class when the state is invalid.

    This is a separate test class because these tests require a different setup/teardown fixture.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self) -> None:
        """Set up and tear down the test environment."""
        self.patcher = patch("os.getpid")
        self.mock_function = self.patcher.start()
        self.mock_function.return_value = _MOCK_PID

        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.exporter = FileExporter(Path(self.temp_file.name))
        self.pid = os.getpid()
        self.maxDiff = None

        yield

        os.unlink(self.temp_file.name)
        self.patcher.stop()

    def test_export_without_initialization(self) -> None:
        """Test that using FileExporter without initialization raises an error."""
        uninitialized_exporter = FileExporter(Path(self.temp_file.name))
        span = Span.create("test_span")
        assert not uninitialized_exporter.ready
        assert not uninitialized_exporter.closed

        with pytest.raises(ExportError, match="Exporter not initialized. Call initialize first."):
            uninitialized_exporter.export_start(span)

    def test_closed_exporter(self) -> None:
        """Test that using FileExporter without initialization raises an error."""
        span = Span.create("test_span")

        assert not self.exporter.ready
        self.exporter.initialize()
        assert self.exporter.ready
        self.exporter.export_start(span)
        assert not self.exporter.closed
        self.exporter.close()
        assert self.exporter.closed
        with pytest.raises(ExportError, match="Exporter is closed and cannot be used anymore."):
            event = span.add_event(Event.create("test_event"))
            self.exporter.export_event(event, span)
