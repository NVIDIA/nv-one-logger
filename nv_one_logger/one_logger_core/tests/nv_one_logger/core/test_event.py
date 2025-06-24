# SPDX-License-Identifier: Apache-2.0
"""Tests for the Event and ErrorEvent classes in one_logger.core."""

import time
import traceback
from unittest.mock import patch

from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import (
    ErrorEvent,
    Event,
    StandardEventAttributeName,
    StandardEventName,
    TelemetryDataError,
)
from nv_one_logger.core.time import TracingTimestamp


class TestEvent:
    """Test cases for Event class."""

    def test_event_creation(self) -> None:
        """Verifies that an Event can be created with a name, timestamp, and optional attributes."""
        timestamp = TracingTimestamp.now()
        attributes = Attributes({"test": 42})

        event = Event(name="test_event", timestamp=timestamp, attributes=attributes)

        assert event.name == "test_event"
        assert event.timestamp == timestamp
        assert event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch, "test": 42})

    def test_event_creation_no_attributes(self) -> None:
        """Verifies that an Event can be created without attributes."""
        timestamp = TracingTimestamp.now()
        event = Event(name="test_event1", timestamp=timestamp)

        assert event.name == "test_event1"
        assert event.timestamp == timestamp
        assert event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch})

    def test_event_creation_with_standard_name(self) -> None:
        """Verifies that an Event can be created with a StandardEventName."""
        timestamp = TracingTimestamp.now()
        event = Event(name=StandardEventName.SPAN_START, timestamp=timestamp)

        assert event.name == StandardEventName.SPAN_START
        assert event.name_str == "span_start"
        assert event.timestamp == timestamp
        assert event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch})

        timestamp = TracingTimestamp.now()
        event = Event(name="test_event2", timestamp=timestamp, attributes=Attributes({"test": 42}))

        assert event.name == "test_event2"
        assert event.timestamp == timestamp
        assert event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch, "test": 42})

    def test_event_create_factory_method(self) -> None:
        """Verifies that the create() factory method creates an event with current timestamp."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            ts_sec = time.time()
            mock_now.return_value = TracingTimestamp.for_timestamp(ts_sec, 100.4)
            event = Event.create(name="test_event")

            assert event.name == "test_event"
            assert event.timestamp.seconds_since_epoch == ts_sec
            assert event.timestamp.perf_counter_seconds == 100.4

    def test_event_create_factory_method_with_custom_timestamp(self) -> None:
        """Verifies that the create() factory method creates an event with a custom timestamp."""
        ts_sec = time.time() - 40
        timestamp = TracingTimestamp.for_timestamp(ts_sec, 100.0)
        attributes = Attributes({"key": "value"})
        event = Event.create(name="test_event", attributes=attributes, timestamp=timestamp)

        assert event.name == "test_event"
        assert event.timestamp == timestamp
        assert event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch, "key": "value"})

    def test_event_serialization(self) -> None:
        """Verifies that to_json() correctly serializes the event."""
        timestamp = TracingTimestamp.now()
        attributes = Attributes()
        attributes.add("number", 42)
        attributes.add("str_list", ["el1", "el2", "el3"])
        attributes.add("float_list", [43.2, 56.1])
        event = Event(name="test_event", timestamp=timestamp, attributes=attributes)

        json_data = event.to_json()

        assert json_data["name"] == "test_event"
        assert json_data["timestamp"] == timestamp.to_json()
        assert json_data["attributes"] == attributes.to_json()

        deserialized_event = Event.from_json(json_data)
        assert deserialized_event == event


class TestErrorEvent:
    """Tests for the ErrorEvent class."""

    def test_error_event_creation(self) -> None:
        """Verifies that an ErrorEvent can be created with error details."""
        timestamp = TracingTimestamp.now()
        error_event = ErrorEvent(
            timestamp=timestamp,
            error_message="Test error",
            exception_type="ValueError",
            exception_message="Invalid value",
            exception_traceback="Traceback...",
        )

        assert error_event.name == StandardEventName.ERROR
        assert error_event.timestamp == timestamp
        assert error_event.error_message == "Test error"
        assert error_event.exception_type == "ValueError"
        assert error_event.exception_message == "Invalid value"
        assert error_event.exception_traceback == "Traceback..."

    def test_error_event_create_with_exception(self) -> None:
        """Verifies that create() with an exception captures all error details."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_event = ErrorEvent.create("Test error", e)
            stack_trace = traceback.format_exc()

            assert error_event.name == StandardEventName.ERROR
            assert error_event.error_message == "Test error"
            assert error_event.exception_type == "ValueError"
            assert error_event.exception_message == "Test error"
            assert error_event.exception_traceback == stack_trace

    def test_error_event_create_with_message_only(self) -> None:
        """Verifies that create() with just an error message works."""
        error_event = ErrorEvent.create("Test error")

        assert error_event.name == StandardEventName.ERROR
        assert error_event.error_message == "Test error"
        assert error_event.exception_type is None
        assert error_event.exception_message is None
        assert error_event.exception_traceback is None

    def test_error_event_serialization(self) -> None:
        """Verifies that to_json() correctly serializes the error event."""
        timestamp = TracingTimestamp.now()
        error_event = ErrorEvent(
            timestamp=timestamp,
            error_message="Test error",
            exception_type="ValueError",
            exception_message="Invalid value",
            exception_traceback="Traceback...",
        )

        json_data = error_event.to_json()

        assert json_data["name"] == StandardEventName.ERROR
        assert json_data["timestamp"] == timestamp.to_json()
        assert json_data["error_message"] == "Test error"
        assert json_data["exception_type"] == "ValueError"
        assert json_data["exception_message"] == "Invalid value"
        assert json_data["exception_traceback"] == "Traceback..."

        deserialized_event = ErrorEvent.from_json(json_data)
        assert deserialized_event == error_event


class TestTelemetryDataIssue:
    """Tests for the TelemetryDataIssue class."""

    def test_telemetry_data_error_creation(self) -> None:
        """Verifies that a TelemetryDataIssue can be created with issue details."""
        timestamp = TracingTimestamp.now()
        issue = TelemetryDataError(
            timestamp=timestamp,
            error_type=TelemetryDataError.ErrorType.NO_TELEMETRY_DATA,
            error_message="Failed to collect telemetry data",
        )

        assert issue.name == StandardEventName.TELEMETRY_DATA_ERROR
        assert issue.timestamp == timestamp
        assert issue.error_type == TelemetryDataError.ErrorType.NO_TELEMETRY_DATA
        assert issue.error_message == "Failed to collect telemetry data"
        assert issue.attributes == Attributes(
            {
                StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch,
                StandardEventAttributeName.ERROR_MESSAGE: "Failed to collect telemetry data",
                StandardEventAttributeName.TELEMETRY_DATA_ERROR_TYPE: TelemetryDataError.ErrorType.NO_TELEMETRY_DATA.value,
            }
        )

    def test_telemetry_data_error_create_factory_method(self) -> None:
        """Verifies that the create() factory method creates an issue with current timestamp."""
        with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
            ts_sec = time.time()
            mock_now.return_value = TracingTimestamp.for_timestamp(ts_sec, 100.4)
            issue = TelemetryDataError.create(
                error_type=TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA,
                error_message="Partial data collected",
            )

            assert issue.name == StandardEventName.TELEMETRY_DATA_ERROR
            assert issue.timestamp.seconds_since_epoch == ts_sec
            assert issue.timestamp.perf_counter_seconds == 100.4
            assert issue.error_type == TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA
            assert issue.error_message == "Partial data collected"

    def test_telemetry_data_error_with_attributes(self) -> None:
        """Verifies that a TelemetryDataIssue can be created with attributes."""
        timestamp = TracingTimestamp.now()
        attributes = Attributes({"error_code": 500, "component": "telemetry_collector"})
        issue = TelemetryDataError(
            timestamp=timestamp,
            error_type=TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA,
            error_message="Data corruption detected",
            attributes=attributes,
        )

        assert issue.name == StandardEventName.TELEMETRY_DATA_ERROR
        assert issue.timestamp == timestamp
        assert issue.error_type == TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA
        assert issue.error_message == "Data corruption detected"
        assert issue.attributes == Attributes(
            {
                StandardEventAttributeName.TIMESTAMP_MSEC: timestamp.milliseconds_since_epoch,
                StandardEventAttributeName.ERROR_MESSAGE: "Data corruption detected",
                StandardEventAttributeName.TELEMETRY_DATA_ERROR_TYPE: TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA.value,
                "error_code": 500,
                "component": "telemetry_collector",
            }
        )

    def test_telemetry_data_error_serialization(self) -> None:
        """Verifies that to_json() correctly serializes the telemetry data issue."""
        timestamp = TracingTimestamp.now()
        attributes = Attributes({"error_code": 500})
        issue = TelemetryDataError(
            timestamp=timestamp,
            error_type=TelemetryDataError.ErrorType.NO_TELEMETRY_DATA,
            error_message="Failed to collect telemetry data",
            attributes=attributes,
        )

        expected_attributes = Attributes()
        expected_attributes.add(StandardEventAttributeName.ERROR_MESSAGE, "Failed to collect telemetry data")
        expected_attributes.add(StandardEventAttributeName.TELEMETRY_DATA_ERROR_TYPE, "no_telemetry_data")
        expected_attributes.add("error_code", 500)
        expected_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, timestamp.milliseconds_since_epoch)

        json_data = issue.to_json()

        assert json_data["name"] == StandardEventName.TELEMETRY_DATA_ERROR
        assert json_data["timestamp"] == timestamp.to_json()
        assert json_data["error_type"] == "no_telemetry_data"
        assert json_data["error_message"] == "Failed to collect telemetry data"
        assert json_data["attributes"] == expected_attributes.to_json()

        deserialized_issue = TelemetryDataError.from_json(json_data)
        assert deserialized_issue == issue

    def test_telemetry_data_error_equality(self) -> None:
        """Verifies that equality comparison works correctly for TelemetryDataIssue."""
        timestamp = TracingTimestamp.now()
        issue1 = TelemetryDataError(
            timestamp=timestamp,
            error_type=TelemetryDataError.ErrorType.NO_TELEMETRY_DATA,
            error_message="Failed to collect telemetry data",
        )
        issue2 = TelemetryDataError(
            timestamp=timestamp,
            error_type=TelemetryDataError.ErrorType.NO_TELEMETRY_DATA,
            error_message="Failed to collect telemetry data",
        )
        issue3 = TelemetryDataError(
            timestamp=timestamp,
            error_type=TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA,
            error_message="Different message",
        )

        assert issue1 == issue2
        assert issue1 != issue3
        assert issue2 != issue3
