# SPDX-License-Identifier: Apache-2.0
import traceback
from enum import Enum
from typing import Any, Dict, Optional, Union

from strenum import StrEnum
from typing_extensions import TypeAlias

from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.time import TracingTimestamp


class StandardEventName(StrEnum):
    """List of standard event names. These events are commonly used for spans regardless of the domain/application type."""

    # An event representing the start of a span.
    SPAN_START = "span_start"
    # An event representing the stop of a span.
    SPAN_STOP = "span_stop"
    # An event representing an error occurring during the execution of the span.
    ERROR = "error"
    # An event representing a telemetry data error.
    TELEMETRY_DATA_ERROR = "telemetry_data_error"


class StandardEventAttributeName(StrEnum):
    """List of standard event attributes. These attributes are commonly used for events regardless of the domain/application type."""

    TIMESTAMP_MSEC = "timestamp_msec"
    ERROR_MESSAGE = "error_message"
    EXCEPTION_TYPE = "exception_type"
    EXCEPTION_MESSAGE = "exception_message"
    EXCEPTION_TRACEBACK = "exception_traceback"
    # The type of a TelemetryDataError.
    TELEMETRY_DATA_ERROR_TYPE = "telemetry_data_error_type"


# The name of an event can be a string or an Enum. Using an Enum is preferred as it avoids issues such
# as typos or inconsistencies in the event name. So instead of passing free form strings, simply define an enum
# that inherits from str and Enum.
EventName: TypeAlias = Union[Enum, str]


class Event:
    """
    A class representing a span event.

    A Span Event represents a meaningful, singular point in time during the Span's duration.
    Each event has a name, timestamp, and a set of attributes.
    See https://opentelemetry.io/docs/concepts/signals/traces/#span-events


    Attributes:
        name: The name of the event.
        timestamp: indicates when the event occurred.
        attributes: Optional dictionary of event attributes.
    """

    def __init__(
        self,
        name: EventName,
        timestamp: TracingTimestamp,
        attributes: Optional[Attributes] = None,
    ):
        """
        Initialize a new Event instance.

        Args:
            name (EventName): The name of the event. See comments on EventName for more details.
            timestamp (TracingTimestamp): The timestamp when the event occurred.
            attributes (Optional[Attributes]): Optional attributes associated with the event.
        """
        assert_that(timestamp, f"Event {name} must have a valid timestamp")
        self._name: EventName = name

        # If you are adding a new field here and you want it to be exported as part of exporting the event,
        # add it to the attributes dictionary too. This helps us avoid adding special code for exporting different fields
        # of the event class.
        # In other words, the fields here are meant to be used in the library code and the attributes are meant to be
        # exported.
        self._timestamp: TracingTimestamp = timestamp

        # We do not allow adding attributes to an event after it is created. This ensures that the event is immutable and
        # reduces the complexity of the export process (e.g., we won't need to deal with the case that an event has changed after being exported).
        self._attributes: Attributes = attributes if attributes is not None else Attributes()
        self._attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, self._timestamp.milliseconds_since_epoch)

    @property
    def name(self) -> EventName:
        """Get the name of the event."""
        return self._name

    @property
    def timestamp(self) -> TracingTimestamp:
        """Get the timestamp of the event."""
        return self._timestamp

    @property
    def attributes(self) -> Attributes:
        """Get the attributes associated with the event."""
        return self._attributes

    @property
    def name_str(self) -> str:
        """Get the name of the event as a string."""
        if isinstance(self.name, Enum):
            return self.name.value
        return self.name

    @classmethod
    def create(cls, name: EventName, attributes: Optional[Attributes] = None, timestamp: Optional[TracingTimestamp] = None) -> "Event":
        """
        Create a new event with the given name and attributes timestamped with the current time.

        Args:
            name (EventName): The name of the event. See comments on EventName for more details.
            attributes (Optional[Attributes]): Optional attributes associated with the event.
        """
        if timestamp is None:
            timestamp = TracingTimestamp.now()
        return cls(name=name, timestamp=timestamp, attributes=attributes)

    def to_json(self) -> Dict[str, Any]:
        """Convert the event to a JSON-compatible dictionary that can be passed to json.dumps.

        Returns:
            dict: a JSON-compatible dictionary representation of the event.
        """
        ret: Dict[str, Any] = {"name": self.name_str, "timestamp": self.timestamp.to_json()}
        if self.attributes:
            ret["attributes"] = self.attributes.to_json()
        return ret

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Event":
        """Create an Event instance from a JSON-compatible dictionary (e.g., the return value of json.loads).

        Args:
            data: Dictionary containing the Event data.

        Returns:
            Event: An Event created from the data.
        """
        assert_that(data, "Event data must be a dictionary")
        assert_that(data.get("name") and data.get("timestamp"), f"Event must have a name and a timestamp: {data}")
        name = data["name"]
        if name == StandardEventName.ERROR:
            return ErrorEvent.from_json(data)
        else:
            return cls(
                name=name,
                timestamp=TracingTimestamp.from_json(data["timestamp"]),
                attributes=Attributes.from_json(data["attributes"]) if "attributes" in data else Attributes(),
            )

    def __eq__(self, other: object) -> bool:
        """Compare this event with another event for equality.

        Args:
            other: The other event to compare with

        Returns:
            bool: True if the events are equal, False otherwise
        """
        if not isinstance(other, Event):
            return False
        return self.name == other.name and self.timestamp == other.timestamp and self.attributes == other.attributes

    def __hash__(self) -> int:
        """Compute a hash value for this event.

        Returns:
            int: A hash value based on the event's name, timestamp, and attributes
        """
        return hash((self.name, self.timestamp, self.attributes))


class ErrorEvent(Event):
    """An event that represents an error in the application and contains an error message or an exception.

    This can be used both for errors that are represented as a string (e.g. an error message)
    as well as errors that are represented as an exception.

    Error events include a few standard attributes:
    - StandardEventAttributes.ERROR_MESSAGE (str): A descriptive message about the error.
    - StandardEventAttributes.EXCEPTION_TYPE (str): The type of the exception if the error was associated with an exception. Not set otherwise.
    - StandardEventAttributes.EXCEPTION_MESSAGE (str): The message from the exception if the error was associated with an exception. Not set otherwise.
    - StandardEventAttributes.EXCEPTION_TRACEBACK (str): The traceback of the exception if the error was associated with an exception and a traceback
        is available. Not set otherwise.
    """

    def __init__(
        self,
        timestamp: TracingTimestamp,
        error_message: str,
        exception_type: Optional[str] = None,
        exception_message: Optional[str] = None,
        exception_traceback: Optional[str] = None,
        attributes: Optional[Attributes] = None,
    ):
        """
        Initialize a new ErrorEvent instance.

        Use the static factory method create() to create an instance instead of this constructor.

        Args:
            timestamp (TracingTimestamp): The timestamp when the error occurred.
            error_message (str): A descriptive message about the error. If the error is based on an exception,
            this can be the same as the exception message or be a more detailed description of the error.
            exception_type (Optional[str]): The type of the exception if applicable
            exception_message (Optional[str]): The message from the exception if applicable
            exception_traceback (Optional[str]): The traceback of the exception if applicable. If
            this is not provided when an exception is provided, the traceback will be captured using traceback.format_exc().
            attributes (Optional[Attributes]): Optional attributes associated with the error event
        """
        assert_that(not exception_type or exception_message, "exception_message must be provided when exception_type is set.")

        # Add a few standard attributes to the event.
        all_attributes = Attributes()
        all_attributes.add(StandardEventAttributeName.ERROR_MESSAGE, error_message)
        if exception_type:
            all_attributes.add(StandardEventAttributeName.EXCEPTION_TYPE, exception_type)
        if exception_message:
            all_attributes.add(StandardEventAttributeName.EXCEPTION_MESSAGE, exception_message)
        if exception_traceback:
            all_attributes.add(StandardEventAttributeName.EXCEPTION_TRACEBACK, exception_traceback)
        if attributes:
            all_attributes.add_attributes(attributes)

        super().__init__(StandardEventName.ERROR, timestamp, all_attributes)
        self.error_message = error_message
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.exception_traceback = exception_traceback

        if exception_type is not None and self.exception_traceback is None:
            self.exception_traceback = traceback.format_exc()

    @classmethod
    def create(  # type: ignore[override]
        cls, error_message: str, exception: Optional[Exception] = None, exception_traceback: Optional[str] = None
    ) -> "ErrorEvent":
        """
        Create a new error event with the given error message, exception and traceback.

        Args:
            error_message (str): A descriptive message about the error. If the error is based on an exception,
            this can be the same as the exception message or be a more detailed description of the error.
            exception (Optional[Exception]): The exception object if applicable
            exception_traceback (Optional[str]): The traceback of the exception if applicable

        Returns:
            ExceptionEvent: A new error event instance
        """
        return cls(
            timestamp=TracingTimestamp.now(),
            error_message=error_message,
            exception_type=exception.__class__.__name__ if exception else None,
            exception_message=str(exception) if exception else None,
            exception_traceback=exception_traceback,
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialize the ErrorEvent to a JSON-compatible dictionary.

        Returns:
            dict: JSON-serializable representation of the ExceptionEvent
        """
        ret = super().to_json()
        ret["error_message"] = self.error_message

        if self.exception_type is not None:
            assert_that(self.exception_traceback is not None, "ErrorEvent must have a traceback if an exception is provided")
            assert_that(
                self.exception_message is not None,
                "ErrorEvent must have an exception message if an exception is provided",
            )
            ret["exception_type"] = self.exception_type
            ret["exception_message"] = self.exception_message
            ret["exception_traceback"] = self.exception_traceback

        return ret

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ErrorEvent":
        """Convert the ErrorEvent to a JSON-compatible dictionary that can be passed to json.dumps.

        Returns:
            dict: a JSON-compatible dictionary representation of the ErrorEvent.
        """
        return cls(
            timestamp=TracingTimestamp.from_json(data["timestamp"]),
            error_message=data["error_message"],
            exception_type=data["exception_type"] if "exception_type" in data else None,
            exception_message=data["exception_message"] if "exception_message" in data else None,
            exception_traceback=data["exception_traceback"] if "exception_traceback" in data else None,
            attributes=Attributes.from_json(data["attributes"]) if "attributes" in data else Attributes(),
        )

    def __eq__(self, other: object) -> bool:
        """Compare this error event with another error event for equality.

        Args:
            other: The other error event to compare with

        Returns:
            bool: True if the error events are equal, False otherwise
        """
        if not isinstance(other, ErrorEvent):
            return False
        return (
            super().__eq__(other)
            and self.error_message == other.error_message
            and self.exception_type == other.exception_type
            and self.exception_message == other.exception_message
            and self.exception_traceback == other.exception_traceback
        )

    def __hash__(self) -> int:
        """Compute a hash value for this error event.

        Returns:
            int: A hash value based on the error event's attributes
        """
        return hash((super().__hash__(), self.error_message, self.exception_type, self.exception_message, self.exception_traceback))


class TelemetryDataError(Event):
    """An event that represents a telemetry data error.

    This can be used to report issues that occurred while collecting telemetry data. Note that the difference between
    TelemetryDataIssue and ErrorEvent is that the latter is for errors in the application itself. In other words,
    ErrorEvent refers to errors that would occur even if OneLogger is disabled whereas TelemetryDataIssue is used to
    represent errors in OneLogger itself or in data collected by OneLogger.

    TelemetryDataError events include a few standard attributes:
    - StandardEventAttributes.TELEMETRY_DATA_ERROR_TYPE (str): The type of the telemetry data error.
    - StandardEventAttributes.ERROR_MESSAGE (str): A descriptive message about the telemetry dataerror.
    """

    class ErrorType(StrEnum):
        """The type of a telemetry data error."""

        # Indicates that no telemetry data could be collected for this execution of the application. For example, when the telemetry code
        # fails and gets disabled before any data is collected. Note that we cannot guarantee that all such cases are detected and reported
        # because even reporting "NO_TELEMETRY_DATA_COLLECTED" may fail. So reporting this condition is done on a best effort basis.
        NO_TELEMETRY_DATA = "no_telemetry_data"

        # Indicates that the telemetry data is incomplete/unreliable/corrupted. For example, if telemetry code encounters
        # a fatal error when some data is already collected or exported to the telemetry backends. This type of error
        # signals the backend that any data collected for the current run of the application is not reliable and should not be
        # used in analysis.
        INCOMPLETE_TELEMETRY_DATA = "incomplete_telemetry_data"

    def __init__(
        self,
        timestamp: TracingTimestamp,
        error_type: ErrorType,
        error_message: str,
        attributes: Optional[Attributes] = None,
    ):
        """
        Initialize a new TelemetryDataIssue instance.

        Use the static factory method create() to create an instance instead of this constructor.

        Args:
            timestamp (TracingTimestamp): The timestamp when the telemetry data error occurred.
            error_type (TelemetryDataErrorType): The type of the telemetry data error.
            error_message (str): A descriptive message about the telemetry data error.
            attributes (Optional[Attributes]): Optional attributes associated with the telemetry data error.
        """
        # Add a few standard attributes to the event.
        all_attributes = Attributes()
        all_attributes.add(StandardEventAttributeName.ERROR_MESSAGE, error_message)
        all_attributes.add(StandardEventAttributeName.TELEMETRY_DATA_ERROR_TYPE, error_type.value)
        if attributes:
            all_attributes.add_attributes(attributes)

        super().__init__(StandardEventName.TELEMETRY_DATA_ERROR, timestamp, all_attributes)
        self.error_type = error_type
        self.error_message = error_message

    @classmethod
    def create(cls, error_type: ErrorType, error_message: str, attributes: Optional[Attributes] = None) -> "TelemetryDataError":  # type: ignore[override]
        """Create a new telemetry data error event.

        Args:
            error_type (TelemetryDataErrorType): The type of the telemetry data error.
            error_message (str): A descriptive message about the telemetry data error.
            attributes (Optional[Attributes]): Optional attributes associated with the telemetry data error.

        Returns:
            TelemetryDataError: A new telemetry data error event instance
        """
        return cls(timestamp=TracingTimestamp.now(), error_type=error_type, error_message=error_message, attributes=attributes)

    def to_json(self) -> Dict[str, Any]:
        """Serialize the TelemetryDataIssue to a JSON-compatible dictionary.

        Returns:
            dict: JSON-serializable representation of the TelemetryDataIssue
        """
        ret = super().to_json()
        ret["error_type"] = self.error_type.value
        ret["error_message"] = self.error_message
        return ret

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TelemetryDataError":
        """Convert the TelemetryDataIssue to a JSON-compatible dictionary that can be passed to json.dumps.

        Returns:
            dict: a JSON-compatible dictionary representation of the TelemetryDataIssue.
        """
        return cls(
            timestamp=TracingTimestamp.from_json(data["timestamp"]),
            error_type=cls.ErrorType(data["error_type"]),
            error_message=data["error_message"],
            attributes=Attributes.from_json(data["attributes"]) if "attributes" in data else Attributes(),
        )

    def __eq__(self, other: object) -> bool:
        """Compare this TelemetryDataIssue event with another TelemetryDataIssue event for equality.

        Args:
            other: The other TelemetryDataError event to compare with

        Returns:
            bool: True if the TelemetryDataError events are equal, False otherwise
        """
        if not isinstance(other, TelemetryDataError):
            return False
        return super().__eq__(other) and self.error_type == other.error_type and self.error_message == other.error_message

    def __hash__(self) -> int:
        """Compute a hash value for this TelemetryDataIssue event.

        Returns:
            int: A hash value based on the TelemetryDataError event's attributes
        """
        return hash((super().__hash__(), self.error_type, self.error_message))
