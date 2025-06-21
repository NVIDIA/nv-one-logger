# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional

from overrides import EnforceOverrides

from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import ErrorEvent, Event
from nv_one_logger.core.span import Span, SpanName
from nv_one_logger.core.time import TracingTimestamp


class Recorder(ABC, EnforceOverrides):
    """An abstract base class for creating and recording of spans and events.

    The application can call the Recorder API to  start/stop spans, add events, or report errors.
    See the "How to use One Logger" section of the README.md file for more information.

    The Recorder is in charge of creating the Span/Event objects and then using one or more exporters
    to send the spans, events, and errors to to one or more backends.
    The recorder decides which exporters to use and when to call them.

    Here are a few examples of what can be done in a Recorder:
    * Filter (not export) certain events (based on attribute values or verbosity).
    # Handle errors occuring while exporting data to each backend.
    * Add new attributes to a span or events or even create new events. For example, if you have a long-running
    span for model training in which multiple "save checkpoint" events are emitted, and you want to keep track
    of the avg, max, and min save times across all the checkpoints, the receiver can keep track of all "save
    checkpoint" events so far and maintain avg, max, and min values of the "duration" attribute of those events
    and then emit a "check point stats update" event periodically.    See the README.md for more information.
    """

    @abstractmethod
    def start(
        self,
        span_name: SpanName,
        span_attributes: Optional[Attributes] = None,
        start_event_attributes: Optional[Attributes] = None,
        start_time: Optional[TracingTimestamp] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """
        Create a new span and record its start event.

        Args:
            name: The name of the span.
            span_attributes: Optional attributes to add to the span.
            start_event_attributes: Optional attributes to add to the start event.
            start_time: The start time of the span; the current timestamp will be used if not specified.
            parent_span: The parent span of the new span. If not specified, the new span will be created as a child of the latest active span
            (or will be a root span if there is no active span).
        Returns:
            The span created.
        """
        pass

    @abstractmethod
    def stop(
        self,
        span: Span,
        stop_event_attributes: Optional[Attributes] = None,
        stop_time: Optional[TracingTimestamp] = None,
    ) -> None:
        """
        Stop the span and record its stop event.

        Args:
            span: The span to stop.
            stop_event_attributes: Optional attributes to add to the stop event.
            stop_time: The stop time of the span; the current timestamp will be used if not specified.
        """
        pass

    @abstractmethod
    def event(self, span: Span, event: Event) -> Event:
        """
        Create a new event and record it immediately.

        The event is added to the last span that was started, unless a span is provided.

        Args:
            span: Span to add the event to.
            event: The event to record.
        Returns:
            The event recorded.
        """
        pass

    @abstractmethod
    def error(
        self,
        span: Span,
        error_message: str,
        exception: Optional[Exception] = None,
    ) -> ErrorEvent:
        """
        Create a new error event with the given error message and exception and record it immediately.

        Use this to report errors in the application. See ErrorEvent for more information.

        The error event is added to the last span that was started, unless a span is provided.

        Args:
            span: Span to add the error event to.
            error_message: The message of the error event.
            exception: The exception of the error event.
        Returns:
            The error event recorded.
        """
        pass

    @abstractmethod
    def telemetry_data_error(self, error_message: str, attributes: Optional[Attributes] = None) -> None:
        """Create a new telemetry data error and record it immediately.

        Use this to report issues that occurred while collecting telemetry data (issues in telemtry code or state rather than the application).
        See TelemetryDataError for more information.

        Args:
            error_message: A descriptive message about the telemetry data error.
            attributes: Optional attributes associated with the event. You can use this to add more information about the error.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the recorder and release any resources.

        All spans that were previously started will be stopped and the stop events recorded at the current time.
        Once the recorder is closed, it cannot be used again.
        """
        pass
