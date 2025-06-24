# SPDX-License-Identifier: Apache-2.0
"""Tests for the Span class in one_logger.core."""

import time
from typing import Dict, Optional

import pytest

from nv_one_logger.core.attributes import Attribute, Attributes, AttributeValue
from nv_one_logger.core.event import Event, StandardEventAttributeName, StandardEventName
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.span import NVTXColor, Span, StandardSpanAttributeName
from nv_one_logger.core.time import TracingTimestamp


class CustomAttributes(Attributes):
    """A custom subclass of Attributes."""

    def __init__(self, dictionary: Optional[Dict[str, AttributeValue]] = None) -> None:
        """Initialize the custom attributes."""
        super().__init__(dictionary)


class TestSpan:
    """Test suite for the Span class."""

    def test_create_span(self) -> None:
        """Test that a span can be created with basic parameters."""
        span_name = "test_span"
        start_time = TracingTimestamp.now()
        span = Span.create(name=span_name, start_time=start_time)

        assert span.name_str == span_name
        assert span.attributes == Attributes()
        assert span.start_event.name == StandardEventName.SPAN_START
        assert span.start_event.timestamp == start_time
        assert len(span.events) == 1
        assert span.active is True
        assert span.start_event
        assert span.start_event.name == StandardEventName.SPAN_START
        assert span.events[0] == span.start_event
        assert span.start_event.timestamp == start_time
        assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: start_time.milliseconds_since_epoch})
        assert span.stop_event is None

    def test_create_span_with_all_optional_args(self) -> None:
        """Test that a span can be created with all optional parameters."""
        span_name = "test_span"
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=time.time() - 10, perf_counter=100.4)
        span_attributes = Attributes(
            {
                "span_attr1": "span_attr1_value",
                "span_attr2": "span_attr2_value",
            }
        )
        start_event_attributes = Attributes(
            {
                "event_attr1": "event_attr1_value",
                "event_attr2": "event_attr2_value",
            }
        )

        span = Span.create(
            name=span_name,
            start_time=start_time,
            span_attributes=span_attributes,
            start_event_attributes=start_event_attributes,
        )

        assert span.name_str == span_name
        assert span.attributes == span_attributes
        assert span.start_event.name == StandardEventName.SPAN_START
        assert span.start_event.timestamp == start_time
        assert len(span.events) == 1
        assert span.active is True
        assert span.start_event
        assert span.start_event.name == StandardEventName.SPAN_START
        assert span.events[0] == span.start_event
        assert span.start_event.timestamp == start_time
        assert span.start_event.attributes == start_event_attributes
        assert span.stop_event is None

    def test_add_event(self) -> None:
        """Test that events can be added to a span."""
        span = Span.create(name="test_span")
        test_event = Event.create(name="test_event")

        span.add_event(test_event)

        assert len(span.events) == 2
        assert span.events[0].name == StandardEventName.SPAN_START
        assert span.events[1] == test_event

    def test_add_event_to_stopped_span_raises_error(self) -> None:
        """Test that adding an event to a stopped span raises an error."""
        span = Span.create(name="test_span")
        span.stop()
        test_event = Event.create(name="test_event")

        with pytest.raises(OneLoggerError, match="Cannot add an event to a span that is not running."):
            span.add_event(test_event)

    def test_add_attribute(self) -> None:
        """Test that attributes can be added to a span."""
        span = Span.create(name="test_span")
        attr_name = "test_attr"
        attr_value = "test_value"

        attribute = span.add_attribute(attr_name, attr_value)

        assert isinstance(attribute, Attribute)
        assert len(span.attributes) == 1
        assert attr_name in span.attributes
        assert span.attributes[attr_name].value == attr_value
        assert len(span.updated_attributes) == 1
        assert span.updated_attributes[attr_name].value == attr_value

    def test_add_attributes(self) -> None:
        """Test that multiple attributes can be added to a span at once."""
        span = Span.create(name="test_span")
        attributes = Attributes({"attr1": "value1", "attr2": "value2"})

        span.add_attributes(attributes)

        assert "attr1" in span.attributes
        assert "attr2" in span.attributes
        assert span.attributes["attr1"].value == "value1"
        assert span.attributes["attr2"].value == "value2"

    def test_add_attributes_with_override(self) -> None:
        """Test that multiple attributes can be added to a span at once."""
        span = Span.create(
            name="test_span",
            span_attributes=Attributes(
                {
                    "span_attr1": "value1",
                    "span_attr2": "value2",
                    "span_attr3": "value3",
                    "span_attr4": "value4",
                }
            ),
        )
        attributes = Attributes(
            {
                "span_attr1": "value1_new",
                "span_attr2": "value2_new",
                "span_attr5": "value5",
            }
        )

        span.add_attributes(attributes)
        span.add_attribute("span_attr4", "value4_new")

        assert span.attributes == Attributes(
            {
                "span_attr1": "value1_new",
                "span_attr2": "value2_new",
                "span_attr3": "value3",
                "span_attr4": "value4_new",
                "span_attr5": "value5",
            }
        )

        assert span.updated_attributes == Attributes(
            {
                "span_attr1": "value1_new",
                "span_attr2": "value2_new",
                "span_attr4": "value4_new",
                "span_attr5": "value5",
            }
        )

    def test_add_attributes_with_custom_attribute_class(self) -> None:
        """Test that adding attributes to a span with custom attributes class works (the subclasss is preserved)."""
        span = Span.create(name="test_span", span_attributes=CustomAttributes())
        span.add_attributes(Attributes({"attr1": "value1"}))
        assert isinstance(span.attributes, CustomAttributes)
        assert isinstance(span.updated_attributes, CustomAttributes)
        assert isinstance(span.attributes, CustomAttributes)
        assert span.attributes == CustomAttributes({"attr1": "value1"})

    def test_stop_span(self) -> None:
        """Test that a span can be stopped and creates a stop event."""
        start_time = TracingTimestamp.for_timestamp(23400.2, 100.2, validate_timestamp=False)
        span = Span.create(name="test_span", start_time=start_time)

        # We intentionally introduce a small inaccuracy for the wall clock (compared to perf counter) to
        # test that the span duration is calculated based on the perf counter and not the wall clock.
        stop_time = TracingTimestamp.for_timestamp(23420.8, 120.9, validate_timestamp=False)
        span.stop(stop_time=stop_time)

        assert not span._timer.running
        assert not span.active
        assert span.stop_event is not None
        assert span.start_event.name == StandardEventName.SPAN_START
        assert span.stop_event.name == StandardEventName.SPAN_STOP
        assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: stop_time.milliseconds_since_epoch})
        assert len(span.events) == 2

    def test_stop_span_with_attributes(self) -> None:
        """Test that a span can be stopped and creates a stop event."""
        start_time = TracingTimestamp.for_timestamp(23400.2, 100.2, validate_timestamp=False)
        span_attributes = Attributes(
            {
                "span_attr1": "value1",
                "span_attr2": "value2",
            }
        )
        start_event_attributes = Attributes(
            {
                "event_attr1": "value1",
                "event_attr2": "value2",
            }
        )

        span = Span.create(
            name="test_span",
            start_time=start_time,
            span_attributes=span_attributes,
            start_event_attributes=start_event_attributes,
        )

        # We intentionally introduce a small inaccuracy for the wall clock (compared to perf counter) to
        # test that the span duration is calculated based on the perf counter and not the wall clock.
        stop_time = TracingTimestamp.for_timestamp(23420.8, 120.9, validate_timestamp=False)
        stop_event_attributes = Attributes({"reason": "test_complete"})
        span.stop(stop_time=stop_time, stop_event_attributes=stop_event_attributes)

        assert not span._timer.running
        assert not span.active
        assert span.attributes == Attributes(
            {
                "span_attr1": "value1",
                "span_attr2": "value2",
                StandardSpanAttributeName.DURATION_MSEC: 20700,  # (120.9 - 100.2)*1000
            }
        )
        assert span.stop_event is not None
        assert span.stop_event.name == StandardEventName.SPAN_STOP
        assert span.stop_event.attributes == stop_event_attributes
        assert span.start_event.name == StandardEventName.SPAN_START
        assert len(span.events) == 2
        assert span.duration_msec() == 20700

    def test_stop_span_twice(self) -> None:
        """Test that stopping a span twice has no effect."""
        span = Span.create(name="test_span")
        span.stop()
        initial_stop_event = span.stop_event

        span.stop()

        assert span.stop_event == initial_stop_event

    def test_span_with_nvtx_color(self) -> None:
        """Test that a span can be created with NVTX color attribute."""
        span = Span.create(name="test_span")
        span.add_attribute(StandardSpanAttributeName.NVTX_COLOR, NVTXColor.BLUE)

        assert span.attributes[StandardSpanAttributeName.NVTX_COLOR].value == NVTXColor.BLUE
