# SPDX-License-Identifier: Apache-2.0
"""Utility functions for testing the training telemetry API."""

from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional
from unittest.mock import MagicMock, Mock

from nv_one_logger.core.event import Event, StandardEventName
from nv_one_logger.core.span import Span
from nv_one_logger.core.time import TracingTimestamp


def advance_time(mock_time: Mock, mock_perf_counter: Mock, seconds: float) -> TracingTimestamp:
    """Advances the mock time by the specified number of seconds and returns a TracingTimestamp corresponding to the new time   ."""
    mock_time.return_value += seconds
    mock_perf_counter.return_value += seconds
    return TracingTimestamp.for_timestamp(timestamp_sec=mock_time.return_value, perf_counter=mock_perf_counter.return_value, validate_timestamp=False)


def assert_only_start_event(span: Span) -> None:
    """Assert that the span has only one event, which is the start event.

    Use for active spans without any additional events.
    """
    assert len(span.events) == 1, "Expected only one event in a span."
    assert span.events[0].name == StandardEventName.SPAN_START, "Expected the start event."


def assert_only_start_stop_event(span: Span, mock_exporter: MagicMock) -> None:
    """Assert that the span has only two events, which are the start and stop events."""
    assert len(span.events) == 2, "Expected only two events in a span."
    assert span.events[0].name == StandardEventName.SPAN_START, "Expected the start event."
    assert span.events[1].name == StandardEventName.SPAN_STOP, "Expected the stop event."


def assert_has_stop_event(span: Span) -> None:
    """Assert that the span has a stop event."""
    assert len(span.events) >= 2
    assert span.events[-1].name == StandardEventName.SPAN_STOP, "Expected the stop event."


def get_non_trivial_events(span: Span) -> List[Event]:
    """Get events other than SPAN_START and SPAN_STOP."""
    assert len(span.events) >= 1
    assert span.events[0].name == StandardEventName.SPAN_START, "Expected the start event."
    if span.events[-1].name == StandardEventName.SPAN_STOP:
        return span.events[1:-1]
    else:
        return span.events[1:]


def assert_exporter_method_call_sequence(mock_exporter: MagicMock, expected_calls: List[Callable[..., Any]]) -> None:
    """Assert that the exporter method calls are in the expected sequence."""
    call_sequence = [call[0] for call in mock_exporter.mock_calls]
    # Remove magic methods
    call_sequence = list(filter(lambda x: not x.startswith("__"), call_sequence))
    assert call_sequence == [call.__name__ for call in expected_calls]


def span_from_export_start(mock_exporter: MagicMock, expected_parent: Optional[Span]) -> Span:
    """Extract the Span object passed to the latest call to export_start."""
    latest_call = mock_exporter.export_start.call_args_list[-1]
    span = latest_call.args[0]
    assert isinstance(span, Span)
    assert span.parent_span is expected_parent
    return span


def span_from_export_stop(mock_exporter: MagicMock) -> Span:
    """Extract the Span object passed to the latest call to export_stop."""
    latest_call = mock_exporter.export_stop.call_args_list[-1]
    span = latest_call.args[0]
    assert isinstance(span, Span)
    return span


def event_from_export_event(mock_exporter: MagicMock, expected_span: Span) -> Event:
    """Extract the Event object passed to the latest call to export_event.

    Use for spans with a single non-trivial event.
    """
    latest_call = mock_exporter.export_event.call_args_list[-1]
    event = latest_call.args[0]
    assert isinstance(event, Event)
    span = latest_call.args[1]
    assert isinstance(span, Span)
    assert expected_span is span

    assert get_non_trivial_events(span)[-1] is event
    return event


def all_events_from_export_event(mock_exporter: MagicMock, expected_span: Span) -> List[Event]:
    """Extract the Event objects passed to all the call to export_event.

    Use for spans with multiple non-trivial event.
    """
    events: List[Event] = []
    for call in mock_exporter.export_event.call_args_list:
        event = call.args[0]
        assert isinstance(event, Event)
        events.append(event)
        span = call.args[1]
        assert isinstance(span, Span)
        assert expected_span is span

    assert get_non_trivial_events(expected_span) == events
    return events


@contextmanager
def assert_no_export(mock_exporter: MagicMock) -> Generator[None, None, None]:
    """Assert that the code inside the context doesn't result in exporting any data."""
    prev_call_count = len(mock_exporter.mock_calls)
    try:
        yield
    finally:
        assert prev_call_count == len(mock_exporter.mock_calls), "Exporter was called when it wasn't supposed to be"
