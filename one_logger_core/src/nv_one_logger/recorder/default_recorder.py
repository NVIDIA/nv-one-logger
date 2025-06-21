# SPDX-License-Identifier: Apache-2.0
import traceback
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from overrides import override  # type: ignore[ancereportUnknownVariableType]

from nv_one_logger.api.recorder import Recorder
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import ErrorEvent, Event, TelemetryDataError
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.core.internal.safe_execution import safely_execute
from nv_one_logger.core.span import Span, SpanName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter

_logger = get_logger(__name__)

# This constant defines the number of failures after which an exporter is disabled. This allows us to avoid unnecessary overhead of calling an exporter
# when it is fatally broken and the data on the backend is not useful due to being incomplete.
_DISABLE_EXPORTER_AFTER_N_FAILURES: int = 5


class ExportCustomizationMode(Enum):
    """The mode of exporting spans (and their associated events and attribytes) to exporters."""

    EXPORT_ALL_SPANS = auto()  # Export all spans and their associated events and attributes to all exporters.
    WHITELIST_SPANS = auto()  # Export only spans that match the whitelist and their associated events and attributes to all exporters.
    BLACKLIST_SPANS = auto()  # Export all spans except those that match the blacklist and their associated events and attributes to all exporters.


@dataclass
class _ExporterState:
    # Number of failed exports so far.
    num_errors: int = 0

    # True means at least some data about the current execution of the application has been successfully sent to this exporter.
    any_data_exported_successfully: bool = False

    # Determines if we force disabled an exporter due to seeing too many consecutive failures.
    force_disabled: bool = False

    # Determines if we have already reported a telemetry data error for this exporter.
    # This variable is used to ensure that we only report telemetry data errors once for each exporter.
    # Note that this could become true either when the exporter encounters an error or when the
    # recorder.telemetry_data_error() method is called by another component of the library.
    telemetry_data_error_reported: bool = False


class DefaultRecorder(Recorder):
    """
    The default recorder implementation of the :class:`one_logger.core.recorder.Recorder` interface.

    This implementation
    - supports exporting to multiple backends.
    - Tracks the parent-child relationship between spans: if a span is started while another span is active, it will be created
      as a child of the active span (unless a parent span is specified explicitly).
    - does NOT filter any spans, events, or attributes nor adds any attributes to the spans or events automatically. However,
      it supports am optional whitelist or blacklist filter for spams to export.
    - It handles errors from exporters differently from other errors. The reason behind this design decision is that
      exporter errors do not necessarily mean that the library has a fatal bug. An intermittent server error could
      cause one exporter to fail while all other exportes may still be working. So for exporter errors, this recorder
        - Catches the error, adds an error log, and records a telemetry data error on the corresponding backend only
          but does not propagate the error to the caller.
        - It disables an exporter after a certain number of failures. This is a heuristic to avoid unnecessary overhead of calling an exporter
          when it is fatally broken and the data on the backend is not useful due to being incomplete.


    It is possible to create a DefaultRecorder with no backends, in which case it will not record anything,
    this is useful if the application is running on thousands of GPUs, in which case the overhead of recording
    from each GPU is too high, and we only want to record from rank zero or a subset of ranks.
    """

    def __init__(
        self,
        exporters: List[Exporter],
        export_customization_mode: ExportCustomizationMode = ExportCustomizationMode.EXPORT_ALL_SPANS,
        span_name_filter: Optional[List[SpanName]] = None,
    ):
        """Initialize the DefaultRecorder with a list of exporters.

        Args:
            exporters: A list of exporters to use.
            export_customization_mode: The mode of exporting spans (and their associated events and attribytes) to exporters.
            span_name_filter: This argument should be interpretted wrt the value of export_customization_mode:
                If export_customization_mode is ExportCustomizationMode.EXPORT_ALL_SPANS, span_name_filter should not be set.
                If export_customization_mode is ExportCustomizationMode.WHITELIST_SPANS, span_name_filter is a list of span names to export (whitelist).
                If export_customization_mode is ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter is a list of span names to not export (blacklist).
        """
        self._exporters = exporters
        # Keep track of the state of each exporter so that we can handle export errors gracefully.
        self._exporter_state: Dict[Exporter, _ExporterState] = {exporter: _ExporterState() for exporter in exporters}

        self._export_customization_mode = export_customization_mode
        self._span_name_filter = span_name_filter

        if export_customization_mode == ExportCustomizationMode.EXPORT_ALL_SPANS:
            assert_that(
                span_name_filter is None, "span_name_filter should not be set when export_customization_mode is ExportCustomizationMode.EXPORT_ALL_SPANS"
            )
        elif export_customization_mode in {ExportCustomizationMode.WHITELIST_SPANS, ExportCustomizationMode.BLACKLIST_SPANS}:
            assert_that(span_name_filter, f"span_name_filter should be a non-empty list when export_customization_mode is {export_customization_mode.name}")
        else:
            raise OneLoggerError(f"Invalid export_customization_mode: {export_customization_mode}")

        # Keeps track of the latest active span in the current context (thread or task). That is, the span that was most recently
        # started and not yet stopped.
        # This is used as the parent span for a newly created span (when no parent span is provided explicitly).
        self._latest_active_span: ContextVar[Optional[Span]] = ContextVar("latest_active_span", default=None)

        self._spans: OrderedDict[uuid.UUID, Span] = OrderedDict()
        if len(self._exporters) > 0:
            _logger.debug(f"Initializing DefaultRecorder with {len(self._exporters)} exporters.")
        else:
            _logger.debug("Initializing DefaultRecorder with no exporters, exporting is disabled")
        for exporter in self._exporters:
            exporter.initialize()

        self._closed = False

    @safely_execute
    @override
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

        Default implementation of Recorder.start() exporting to all exporters of the recorder.
        See the docstrings of the Recorder interface for more details.
        """
        assert_that(not self._closed, "Recorder is already closed.")
        span = Span.create(
            name=span_name,
            parent_span=parent_span if parent_span else self._latest_active_span.get(),
            span_attributes=span_attributes,
            start_event_attributes=start_event_attributes,
            start_time=start_time,
        )
        self._spans[span.id] = span
        self._latest_active_span.set(span)
        if self._should_export_span(span):
            for exporter in self._exporters:
                self._execute_export(exporter, exporter.export_start, span)
        return span

    @safely_execute
    @override
    def stop(
        self,
        span: Span,
        stop_event_attributes: Optional[Attributes] = None,
        stop_time: Optional[TracingTimestamp] = None,
    ) -> None:
        """Stop the span and record its stop event.

        Default implementation of Recorder.stop() exporting to all exporters of the recorder.
        See the docstrings of the Recorder interface for more details.
        """
        assert_that(not self._closed, "Recorder is already closed.")
        assert_that(span, "No span to stop")
        try:
            self._spans.pop(span.id)
        except KeyError:
            raise OneLoggerError(f"Span {span.id} not found")

        span.stop(stop_event_attributes, stop_time)

        # If the span that is being stopped is the latest active span, we need to update the latest active span to the parent of the span.
        if self._latest_active_span.get() and span.id == self._latest_active_span.get().id:
            # Normally, that parent of the current active span should be active, but let's err on the side of caution and check.
            if span.parent_span and not span.parent_span.active:
                _logger.error(
                    f"The parent of span {span.name_str} (id={span.id}) is stopped before its child span. "
                    f"The parent span is {span.parent_span.name_str} (id={span.parent_span.id})"
                )
            new_latest_active_span = span.parent_span
            while new_latest_active_span and not new_latest_active_span.active:
                new_latest_active_span = new_latest_active_span.parent_span
            self._latest_active_span.set(new_latest_active_span)

        if self._should_export_span(span):
            for exporter in self._exporters:
                self._execute_export(exporter, exporter.export_stop, span)

    @safely_execute
    @override
    def event(self, span: Span, event: Event) -> Event:
        """Add an event to a span and record it.

        Default implementation of Recorder.event() exporting to all exporters of the recorder.
        See the docstrings of the Recorder interface for more details.
        """
        assert_that(not self._closed, "Recorder is already closed.")
        assert_that(span, "No span to add event to")
        span.add_event(event)

        if self._should_export_span(span):
            for exporter in self._exporters:
                self._execute_export(exporter, exporter.export_event, event, span)
        return event

    @safely_execute
    @override
    def error(
        self,
        span: Span,
        error_message: str,
        exception: Optional[Exception] = None,
    ) -> ErrorEvent:
        """Add an error to a span and record it.

        Default implementation of Recorder.error() exporting to all exporters of the recorder.
        See the docstrings of the Recorder interface for more details.
        """
        assert_that(not self._closed, "Recorder is already closed.")
        assert_that(span, "No span to add error to")

        event = ErrorEvent.create(
            error_message=error_message,
            exception=exception,
            exception_traceback=traceback.format_exc() if exception else None,
        )
        span.add_event(event)
        if self._should_export_span(span):
            for exporter in self._exporters:
                self._execute_export(exporter, exporter.export_error, event, span)
        return event

    @safely_execute
    @override
    def telemetry_data_error(self, error_message: str, attributes: Optional[Attributes] = None) -> None:
        """Report a telemetry data error.

        Default implementation of Recorder.telemetry_data_issue() exporting the telemetry data error to all exporters of the recorder.
        See the docstrings of the Recorder interface for more details.
        """
        assert_that(not self._closed, "Recorder is already closed.")

        for exporter in self._exporters:
            self._export_telemetry_data_error(exporter, error_message, attributes)

    def _export_telemetry_data_error(self, exporter: Exporter, error_message: str, attributes: Optional[Attributes] = None) -> None:
        exporter_state = self._exporter_state[exporter]
        if exporter_state.telemetry_data_error_reported:
            return
        exporter_state.telemetry_data_error_reported = True
        error = TelemetryDataError.create(
            error_type=self._determine_telemetry_data_error_type(exporter_state), error_message=error_message, attributes=attributes
        )
        self._execute_export(exporter, exporter.export_telemetry_data_error, error)

    def _determine_telemetry_data_error_type(self, exporter_state: _ExporterState) -> TelemetryDataError.ErrorType:
        if exporter_state.any_data_exported_successfully:
            return TelemetryDataError.ErrorType.INCOMPLETE_TELEMETRY_DATA
        return TelemetryDataError.ErrorType.NO_TELEMETRY_DATA

    @safely_execute
    @override
    def close(self) -> None:
        """Close the recorder.

        Default implementation of Recorder.close() closing all exporters of the recorder.
        See the docstrings of the Recorder interface for more details.
        """
        # Whatever is left in the _spans dict, is an active span, we need to stop it.
        for span in reversed(self._spans.values()):
            self.stop(span)
        self._spans.clear()

        for exporter in self._exporters:
            try:
                exporter.close()
            except Exception as e:
                _logger.error(f"Could not close the exporter {exporter.__class__.__name__}: {e}")

        self._closed = True

    def get_active_spans_by_name(self, span_name: SpanName) -> List[Span]:
        """Get all activespans with the given name.

        Args:
            span_name: The name of the spans to find.

        Returns:
            List[Span]: A list of all spans with the given name. Note that in the most general case, nested spans can have the same name
            (but they will have different IDs and different parents). For example, if you define a span to cover the body of function foo()
            and foo() calls itself (directly or indirectly), while this function call is in progress,you will have two active spans with the same name.
        """
        return [span for span in self._spans.values() if span.name == span_name]

    def _should_export_span(self, span: Span) -> bool:
        """Check if the span should be exported based on the export customization mode and the span name filter."""
        if self._export_customization_mode == ExportCustomizationMode.EXPORT_ALL_SPANS:
            return True
        elif self._export_customization_mode == ExportCustomizationMode.WHITELIST_SPANS and span.name in self._span_name_filter:
            return True
        elif self._export_customization_mode == ExportCustomizationMode.BLACKLIST_SPANS and span.name not in self._span_name_filter:
            return True
        return False

    def _execute_export(self, exporter: Exporter, method: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Execute the export method of the exporter.

        This method is responsible for executing export_xxx methods of the exporter and handles any errors that occur during the export
        in a specific way (see the class docstrings for more info).
        """
        if self._exporter_state[exporter].force_disabled:
            return
        exporter_state = self._exporter_state[exporter]
        try:
            method(*args, **kwargs)
            exporter_state.any_data_exported_successfully = True
        except Exception as e:
            exporter_state.num_errors += 1
            error_message = f"Error exporting data via exporter {exporter.__class__.__name__}: {e}"
            _logger.error(error_message)
            try:
                # call the exporter directly (not through _execute_export) as this is a special export call
                # to report an export issue. So we treat it differently (e.g., its success doesn't change
                # the any_data_exported_successfully flag)
                if not exporter_state.telemetry_data_error_reported:
                    exporter_state.telemetry_data_error_reported = True
                    exporter.export_telemetry_data_error(
                        TelemetryDataError.create(error_type=self._determine_telemetry_data_error_type(exporter_state), error_message=error_message)
                    )

            except Exception as e2:
                _logger.error(f"Could not export telemetry data error for exporter {exporter.__class__.__name__}: {e2}")
            # Note that we are not re-raising the original error. See class docstrings for more details.
        finally:
            if exporter_state.num_errors >= _DISABLE_EXPORTER_AFTER_N_FAILURES:
                error_message = f"Exporter {exporter.__class__.__name__} has been force disabled due to too many consecutive failures."
                _logger.error(error_message)
                exporter_state.force_disabled = True
