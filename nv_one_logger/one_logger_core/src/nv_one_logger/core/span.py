# SPDX-License-Identifier: Apache-2.0
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from strenum import StrEnum
from typing_extensions import TypeAlias

from nv_one_logger.core.attributes import Attribute, Attributes, AttributeValue
from nv_one_logger.core.event import Event, StandardEventName
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.time import Timer, TracingTimestamp


class StandardSpanName(StrEnum):
    """List of span names that are commonly used (across all domains) for spans."""

    # This span corresponds to the entire execution of the application (in other words, the main() function).
    APPLICATION = "application"


class StandardSpanAttributeName(StrEnum):
    """
    List of attributes that are commonly used for spans.

    Only include attributes that are generic (not specific to any domain) and are applicable to all spans.
    For domain specific attributes, you can define your own class that inherits from both str and Enum (preferred) or
    simply pass the attribute name as a string.
    """

    # The duration of a stopped span in milliseconds. This attribute is automatically added when the span is stopped.
    DURATION_MSEC = "duration_msec"

    #  The color assigned to the NVTX marker for this span. Use NVTXColor for the values.
    NVTX_COLOR = "nvtx_color"


class NVTXColor(StrEnum):
    """The color assigned to an NVTX marker."""

    UNSET = "unset"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    PURPLE = "purple"
    RAPIDS = "rapids"
    CYAN = "cyan"
    RED = "red"
    WHITE = "white"
    DARK_GREEN = "darkgreen"
    ORANGE = "orange"


# The name of a span can be a string or an Enum. Using an Enum is preferred as it avoids issues such
# as typos or inconsistencies in the event name. So instead of passing free form strings, simply define an enum
# that inherits from str and Enum.
SpanName: TypeAlias = Union[Enum, str]


class Span:
    """A class representing a span in a distributed tracing system.

    A span represents a unit of work or operation. It can contain events, and can have attributes associated with it.
    Spans are used to track the execution of operations and their relationships in a distributed system.
    See https://opentelemetry.io/docs/concepts/signals/traces/#spans.

    A span can have a set of events as well as attributes associated with it.
    In our library, we ensure that a start event is created when the span is created and a stop event when the span is stopped.
    """

    def __init__(
        self,
        id: uuid.UUID,
        name: SpanName,
        start_event: Event,
        parent_span: Optional["Span"] = None,
        span_attributes: Optional[Attributes] = None,
    ):
        """Initialize a new Span instance.

        Note: Prefer using the static create() function instead of directly creating an instance of this class.

        Args:
            id: A unique identifier for the span.
            name: The name of the span.
            start_event: The event that marks the beginning of the span.
            span_attributes: Optional attributes to add to the span. If you are passing a subclass of Attributes, we ensure that the subclass typeis preserved
            even if you add more attributes later.
        """
        assert_that(start_event, f"Span {name} does not have a start event.")
        assert_that(
            start_event and start_event.timestamp and start_event.timestamp.perf_counter_seconds and start_event.timestamp.perf_counter_seconds,
            f"Span {name} must have a start event with a valid timestamp but got {start_event.timestamp}",
        )

        self._id: uuid.UUID = id
        self._name: SpanName = name
        self._parent_span: Optional["Span"] = parent_span
        self._events: List[Event] = [start_event]

        # We keep track of which attributes were set upon the creation of the span vs the ones that were added or changed after the span was created.
        # This is useful in cases where we want to export the span and its initial attributes upon creation and then upon completion of the span,
        # we want to only export any new or changed attributes.
        if span_attributes is None:  # Don't change this to "not span_attributes". We don't want to change the type (subclass type) even if the dict is empty.
            span_attributes = Attributes()
        self._initial_attributes: Attributes = span_attributes
        # Let's make sure the type of the _updated_attributes field matches that of _initial_attributes (e.g., the same subclass of Attributes).
        self._updated_attributes: Attributes = type(self._initial_attributes)({})

        self._timer = Timer()
        self._timer.start(start_time=start_event.timestamp)

    @property
    def id(self) -> uuid.UUID:
        """Get the unique identifier of the span."""
        return self._id

    @property
    def name(self) -> SpanName:
        """Get the name of the span."""
        return self._name

    @property
    def parent_span(self) -> Optional["Span"]:
        """Get the parent span of the span or None if the span is a root span."""
        return self._parent_span

    @property
    def events(self) -> List[Event]:
        """Get the list of events associated with the span."""
        return self._events

    @property
    def attributes(self) -> Attributes:
        """Get the attributes associated with the span.

        This includes the attributes set during creation as well as any updated attributes.
        """
        # Make sure we preserve the subclass of the attributes.
        return type(self._initial_attributes).merge(self._initial_attributes, self._updated_attributes)

    @property
    def updated_attributes(self) -> Attributes:
        """Get the attributes associated with the span that were added or changed  after the span was created."""
        return self._updated_attributes

    @property
    def name_str(self) -> str:
        """Get the name of the span as a string."""
        return str(self.name)

    @property
    def start_event(self) -> Event:
        """Get the start event of the span."""
        assert_that(len(self.events) and self.events[0].name == StandardEventName.SPAN_START, "Span must have a start event.")
        return self.events[0]

    @property
    def stop_event(self) -> Optional[Event]:
        """Returns the stop event if the span has been stopped, otherwise returns None."""
        if not self._timer.running:
            assert_that(
                len(self.events) > 1 and self.events[-1].name == StandardEventName.SPAN_STOP,
                "Span must have a stop event if it has been stopped.",
            )
            return self.events[-1]
        return None

    @property
    def active(self) -> bool:
        """Check if the span is still active."""
        assert_that(
            self._timer.running or (len(self.events) > 1 and self.events[-1].name == StandardEventName.SPAN_STOP),
            f"inconsistent span state: time running: {self._timer.running}, events: {self.events}",
        )
        return self._timer.running

    @classmethod
    def create(
        cls,
        name: SpanName,
        parent_span: Optional["Span"] = None,
        span_attributes: Optional[Attributes] = None,
        start_event_attributes: Optional[Attributes] = None,
        start_time: Optional[TracingTimestamp] = None,
    ) -> "Span":
        """Create a new span with the given parameters.

        Args:
            name: The name of the span.
            parent_span: The parent span of the new span. If not specified, the new span will be a root span.
            start_time: The start time of the span; the current timestamp will be used if not specified.
            span_attributes: Optional attributes to add to the span.
            start_event_attributes: Optional attributes to add to the start event.
            start_time: The start time of the span; the current time will be used if not specified.

        Returns:
            A new Span instance initialized with the provided parameters.
        """
        if not start_time:
            start_time = TracingTimestamp.now()
        start_event = Event(name=StandardEventName.SPAN_START, attributes=start_event_attributes, timestamp=start_time)
        return cls(id=uuid.uuid4(), name=name, start_event=start_event, parent_span=parent_span, span_attributes=span_attributes)

    def stop(self, stop_event_attributes: Optional[Attributes] = None, stop_time: Optional[TracingTimestamp] = None) -> None:
        """Stop the span.

        Args:
            stop_event_attributes: Optional attributes to add to the stop event.
            stop_time: The stop time of the span; the current timestamp will be used if not specified.
        """
        if not self._timer.running:
            return  # already stopped
        if not stop_time:
            stop_time = TracingTimestamp.now()
        stop_event = Event(name=StandardEventName.SPAN_STOP, attributes=stop_event_attributes, timestamp=stop_time)
        self.add_event(stop_event)
        self._timer.stop(reset=True)
        self.add_attribute(StandardSpanAttributeName.DURATION_MSEC, _get_complete_span_duration_msec(self))

    def add_event(self, event: Event) -> Event:
        """Add an event to the span."""
        assert_that(self._timer.running, "Cannot add an event to a span that is not running.")
        self.events.append(event)
        return event

    def add_attribute(self, name: str, value: AttributeValue) -> Attribute:
        """
        Add an attribute to the span.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.

        Returns:
            The attribute.
        """
        self._updated_attributes[name] = Attribute(name=name, value=value)
        return self._updated_attributes[name]

    def add_attributes(self, attributes: Attributes) -> None:
        """Add a dictionary of attributes to the span."""
        self._updated_attributes.update(attributes)

    def to_json(self) -> Dict[str, Any]:
        """Convert the span to a JSON-compatible dictionary that can be passed to json.dumps.

        Returns:
            dict: a JSON-compatible dictionary representation of the span.
        """
        ret: Dict[str, Any] = {"name": self.name_str, "id": self.id, "events": [event.to_json() for event in self.events]}
        if self.attributes:
            ret["attributes"] = self.attributes.to_json()
        return ret

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Span":
        """Create a Span instance from a JSON-compatible dictionary (e.g., the return value of json.loads).

        Args:
            data: Dictionary containing the Span data.

        Returns:
            Event: A Span created from the data.
        """
        assert_that(data, "Span data must be a dictionary")
        assert_that(data.get("name") and data.get("id"), f"Span must have a name and an id: {data}")
        assert_that(
            data.get("events") and isinstance(data.get("events"), list) and len(data.get("events")) > 0,
            f"Span must have a start event: {data}",
        )
        events = [Event.from_json(event) for event in data["events"]]
        span = cls(
            id=data["id"],
            name=data["name"],
            start_event=events[0],
            span_attributes=Attributes.from_json(data["attributes"]) if "attributes" in data else Attributes(),
        )
        for event in events[1:]:
            span.add_event(event)
        return span

    def duration_msec(self) -> int:
        """Get the duration of the span in milliseconds.

        This method can be called only after the span is stopped.
        """
        assert_that(not self.active and self.stop_event, f"Span must be running or stopped and have a stop event {self}.")
        assert_that(StandardSpanAttributeName.DURATION_MSEC in self.attributes, f"Span must have a duration attribute: {self}.")
        duration = self.attributes.get(StandardSpanAttributeName.DURATION_MSEC).value
        assert_that(isinstance(duration, int), f"Span must have a duration attribute: {self}.")
        return cast(int, duration)


def _get_complete_span_duration_msec(span: Span) -> int:
    """
    Get the duration of a complete span.

    Args:
        span: The span to get the duration of. This span must have a stop event.

    Returns:
        The duration of the span in milliseconds.
    """
    assert_that(span.stop_event, "Span must be stopped and have a stop event.")
    duration_sec = span.stop_event.timestamp.perf_counter_seconds - span.start_event.timestamp.perf_counter_seconds  # type: ignore[union-attr]
    return int(duration_sec * 1000)
