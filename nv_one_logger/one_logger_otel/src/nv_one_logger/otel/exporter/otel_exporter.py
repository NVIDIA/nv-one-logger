# SPDX-License-Identifier: Apache-2.0
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Dict, Optional, Tuple

from nv_one_logger.core.attributes import Attributes, AttributeValue
from nv_one_logger.core.event import ErrorEvent, Event, StandardEventName, TelemetryDataError
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.span import Span
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import BaseExporter, ExportError
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util import types as otel_types
from overrides import override  # type: ignore[ancereportUnknownVariableType]

_ONE_LOGGER_SPAN_ID_KEY = "one_logger_span_id"


class OTelExporter(BaseExporter):
    """Exporter implementation that send spans and events to an OTEL receiver."""

    def __init__(self, otel_span_exporter: SpanExporter) -> None:
        """Construct the exporter.

        Args:
            otel_span_exporter: The OTEL span exporter (opentelemetry.sdk.trace.export.SpanExporter) to use.

        Raises:
            ExportError: If there is an error during the export operation.
        """
        super().__init__()
        # Maps a span id to a tuple of the OTEL span and the context.
        self._trace_spans_with_contexts: Dict[uuid.UUID, Tuple[trace.Span, context_api.Context]] = {}
        self._otel_span_exporter = otel_span_exporter

    @override
    def initialize(self) -> None:  # type: ignore[unused-parameter]
        """Initialize the exporter.

        Raises:
            ExportError: If there is an error during the export operation.
        """
        super().initialize()
        try:
            self._provider = TracerProvider()
            self._processor = BatchSpanProcessor(self._otel_span_exporter)
            self._provider.add_span_processor(self._processor)
            self._tracer = self._provider.get_tracer("one_logger")
        except Exception as e:
            raise ExportError(f"Error initializing the exporter: {e}") from e

    @override
    def export_start(self, span: Span) -> None:
        """Export a newly started span along with its current attributes and its start event.

        Args:
            span: The span to export. This span can be still in progress.

        Raises:
            ExportError: If there is an error during the export operation.
        """
        super().export_start(span)
        start_time_ns = self._convert_timestamp(span.start_event.timestamp)
        attributes = self._convert_attributes(span.attributes)

        try:
            parent_context: Optional[context_api.Context] = None
            if span.parent_span:
                assert_that(self._trace_spans_with_contexts.get(span.parent_span.id), f"Cannot find the parent span {span.parent_span.id} of span {span.id}")
                _, parent_context = self._trace_spans_with_contexts[span.parent_span.id]
            trace_span = self._tracer.start_span(
                name=span.name_str,
                context=parent_context,
                start_time=start_time_ns,
                attributes=attributes,
            )
            # OTEL has its own ID for spans. We need to store the OneLogger span ID as an attribute.
            trace_span.set_attribute(_ONE_LOGGER_SPAN_ID_KEY, span.id.int)

            context = set_span_in_context(trace_span)
            context_api.attach(context)
        except Exception as e:
            raise ExportError(f"Error exporting span {span.id}: {e}") from e

        self._trace_spans_with_contexts[span.id] = (trace_span, context)

        self.export_event(span.start_event, span)

    @override
    def export_stop(self, span: Span) -> None:
        """Export a stopped/finished span along with its current attributes and its stop event.

        Args:
            span: The span to export. This span must be already stopped.

           This span must be already stopped.

        Raises:
            ExportError: If there is an error during the export operation.
        """
        super().export_stop(span)
        if span.id not in self._trace_spans_with_contexts:
            raise ExportError(f"Span {span.id} not found")
        if span.stop_event is None:
            raise ExportError(f"Span {span.id} has no stop event")

        trace_span, _ = self._trace_spans_with_contexts[span.id]
        # Update the OTEL span attributes (in case the attributes have changed since the creation of the span.)
        span_attributes = self._convert_attributes(span.attributes)

        try:
            if span_attributes:
                trace_span.set_attributes(span_attributes)
            self.export_event(span.stop_event, span)
            trace_span.end(end_time=self._convert_timestamp(span.stop_event.timestamp))
        except Exception as e:
            raise ExportError(f"Error exporting span {span.id}: {e}") from e
        finally:
            del self._trace_spans_with_contexts[span.id]

    @override
    def export_event(self, event: Event, span: Span) -> None:
        """Export an event that occurred for an active span.

        Note: you do not need to call this method for start and stop events as they are exported in the export_start and export_stop methods.
        Args:
            event: The event to export.
            span: The span that the event belongs to. This span must be still active.

        Raises:
            ExportError: If there is an error during the export operation.
        """
        super().export_event(event, span)
        if span.id not in self._trace_spans_with_contexts:
            raise ExportError(f"Span {span.id} not found")

        # There is an API to add an event directly to the parent span, but OTEL curretly
        # does not report events until the span ends.
        # For long-lived spans, this is a problem as the lifetime of a span could be the entire application
        # runtime or training loop time.  This issue and a request to provide an API to report events for
        # for an active span incrementally is discussed multiple times without any clear resolution:
        # https://github.com/open-telemetry/opentelemetry-specification/discussions/3732
        # https://github.com/open-telemetry/opentelemetry-specification/issues/373
        # As a workaround, we report the event as a new (fake) span within the same context. The fake span
        # is reported as a child of the real span with a runtime of zero.

        parent_span, parent_context = self._trace_spans_with_contexts[span.id]
        attributes = self._convert_attributes(event.attributes)
        event_time_ns = self._convert_timestamp(event.timestamp)

        try:
            fake_event_span = self._tracer.start_span(
                name=event.name_str + f"_event_for_{span.name_str}",
                context=parent_context,
                start_time=event_time_ns,
                attributes=attributes,
            )
            fake_event_span.end(end_time=event_time_ns)
            parent_span.add_event(event.name_str, attributes=self._convert_attributes(event.attributes), timestamp=event_time_ns)
        except Exception as e:
            raise ExportError(f"Error exporting event {event.name_str}: {e}") from e

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Export an error event that occurred for an active span.

        Args:
            event: The error event to export.
            span: The span that the error event belongs to. This span must be still active.

        Raises:
            ExportError: If there is an error during the export operation.
        """
        super().export_error(event, span)
        if span.id not in self._trace_spans_with_contexts:
            raise ExportError(f"Span {span.id} not found")

        parent_span, parent_context = self._trace_spans_with_contexts[span.id]

        event_time_ns = self._convert_timestamp(event.timestamp)

        # See the comment in export_event for the reasoning behind the fake event span.
        try:
            fake_event_span = self._tracer.start_span(
                name=event.name_str + f"_for_{span.name_str}",
                context=parent_context,
                start_time=event_time_ns,
                attributes=self._convert_attributes(event.attributes),
            )
            fake_event_span.set_status(Status(status_code=StatusCode.ERROR, description=event.error_message))
            fake_event_span.end(end_time=event_time_ns)
            parent_span.add_event(event.name_str, attributes=self._convert_attributes(event.attributes), timestamp=event_time_ns)
            parent_span.set_status(Status(status_code=StatusCode.ERROR, description=event.error_message))
        except Exception as e:
            raise ExportError(f"Error exporting error {event.name_str}: {e}") from e

    @override
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Export a telemetry data error."""
        super().export_telemetry_data_error(error)

        attributes = self._convert_attributes(error.attributes)
        event_time_ns = self._convert_timestamp(error.timestamp)
        # We assign the error to a  root span. We could also make it an event of the APPLICATION span but
        # we don't want to make an assumption that such a span exists. So we just create a dedicated/fake span
        # for reporting the telemetry data error event.
        try:
            fake_event_span = self._tracer.start_span(
                name="span_for_" + StandardEventName.TELEMETRY_DATA_ERROR.value,
                start_time=event_time_ns,
            )
            fake_event_span.set_status(Status(status_code=StatusCode.ERROR, description=error.error_message))
            fake_event_span.add_event(StandardEventName.TELEMETRY_DATA_ERROR.value, attributes=attributes, timestamp=event_time_ns)
            fake_event_span.end(end_time=event_time_ns)
        except Exception as e:
            raise ExportError(f"Error exporting telemetry data error {error.name_str}: {e}") from e

    @override
    def close(self) -> None:
        """Shut down the exporter.

        Use this method to release any resources held by the exporter or flush any pending data that is not yet exported.
        Raises:
            ExportError: If there is an error during the export operation.
        """
        try:
            self._provider.shutdown()
        except Exception as e:
            raise ExportError(f"Error closing the exporter: {e}") from e
        super().close()

    def _convert_attributes(self, attributes: Attributes) -> Optional[otel_types.Attributes]:
        """Convert attributes to OTEL supported attributetypes."""

        def _is_native_otel_attribute(value: AttributeValue) -> bool:
            # Check against OTEL supported types (scalars and sequences of primitives)
            # This code is based on the "from opentelemetry.util.types.AttributeValue"
            # type alias, which we cannot use directly with is_instance as it includes
            # generic types.
            is_supported_scalar = isinstance(value, (str, bool, int, float))
            is_supported_sequence = False
            if isinstance(value, Sequence) and not isinstance(value, str):  # str is Sequence, exclude it
                if value:  # If sequence is not empty
                    first_elem_type = type(value[0])
                    if first_elem_type in (str, bool, int, float):
                        # Check if all elements have the same primitive type
                        is_supported_sequence = all(isinstance(x, first_elem_type) for x in value)
                else:  # Empty sequence is supported
                    is_supported_sequence = True

            return is_supported_scalar or is_supported_sequence

        if not attributes:
            return None
        otel_attributes: Dict[str, otel_types.AttributeValue] = {}

        for name, attribute in attributes.items():
            # If the attribute name or value is not natively supported by OTEL, we convert it to a string.
            name_str = name.value if isinstance(name, Enum) else name
            otel_attributes[name_str] = attribute.value if _is_native_otel_attribute(attribute.value) else str(attribute.value)
        return otel_attributes

    def _convert_timestamp(self, timestamp: TracingTimestamp) -> int:
        """Convert the time represented by TracingTimestamp to a nanoseconds since epoch.

        This is needed because OTEL expects timestamps in nanoseconds since epoch.
        """
        return int(timestamp.seconds_since_epoch * 1_000_000_000)
