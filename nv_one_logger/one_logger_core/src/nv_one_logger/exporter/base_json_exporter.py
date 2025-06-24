# SPDX-License-Identifier: Apache-2.0
import os
from abc import abstractmethod
from threading import RLock
from typing import Any, Dict, Optional

from overrides import override  # type: ignore[ancereportUnknownVariableType]

from nv_one_logger.core.event import ErrorEvent, Event, TelemetryDataError
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.core.span import Span
from nv_one_logger.exporter.exporter import BaseExporter
from nv_one_logger.exporter.internal.record_type import RecordType

_logger = get_logger(__name__)


class BaseJsonExporter(BaseExporter):
    """A base class that exports spans and events and their attributes as JSON.

    Events are formatted as key=value pairs with hierarchical keys separated by dots.
    Each implementation can choose where to write/send the JSON data.
    """

    def __init__(self) -> None:
        """Initialize the BaseJsonExporter."""
        super().__init__()
        self._record_count: int = 0
        self._lock: RLock = RLock()
        self._pid: int = os.getpid()

        # This keeps track of the last span that was started but not yet written to the log.
        # This allows us to do an ioptimization: write a single record for a span that
        # has no children spans.
        # We only allow one span to be open but not written to the log at a time.
        self._last_start_span: Optional[Span] = None

    @override
    def initialize(self) -> None:
        """Initialize the BaseJsonExporter.

        Implementation of Exporter.initialize() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        super().initialize()
        self._last_start_span = None

    @override
    def export_start(self, span: Span) -> None:
        """Start the span and record its start event.

        Implementation of Exporter.export_start() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        super().export_start(span)

        if self._last_start_span is not None:
            self._record_span(self._last_start_span, RecordType.START)
        self._last_start_span = span

    @override
    def export_stop(self, span: Span) -> None:
        """Stop the span and record its stop event.

        Implementation of Exporter.export_stop() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        super().export_stop(span)

        if span.stop_event is None:
            raise OneLoggerError(f"Span {span.id} has no stop event")
        if self._last_start_span:
            if self._last_start_span.id == span.id:
                # Use a single record for a complete span
                # rather than two consecutive start and stop events.
                self._record_span(span, RecordType.COMPLETE)
                self._last_start_span = None
                return
            else:
                # record the previous span before recording the unrelated stop event.
                self._record_span(self._last_start_span, RecordType.START)
                self._last_start_span = None

        # record the stop event
        self._record_span(span, RecordType.STOP)

    @override
    def export_event(self, event: Event, span: Span) -> None:
        """Add an event to a span and record it.

        Implementation of Exporter.export_event() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        super().export_event(event, span)

        if self._last_start_span:
            self._record_span(self._last_start_span, RecordType.START)
            self._last_start_span = None
        json_dict = {
            "type": RecordType.EVENT.value,
            "span_id": str(span.id),
            "event": self._event_data_to_json(event),
        }
        self._export_json_dict(json_dict)

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Add an error to a span and record it.

        Implementation of Exporter.export_error() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        super().export_error(event, span)

        if self._last_start_span:
            self._record_span(self._last_start_span, RecordType.START)
            self._last_start_span = None
        json_dict = {
            "type": RecordType.APPLICATION_ERROR.value,
            "span_id": str(span.id),
            "event": self._event_data_to_json(event),
        }
        self._export_json_dict(json_dict)

    @override
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Report a telemetry data error.

        Implementation of Exporter.export_telemetry_data_error() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        super().export_telemetry_data_error(error)

        json_dict = {
            "type": RecordType.TELEMETRY_DATA_ERROR.value,
            "error": self._event_data_to_json(error),
        }
        self._export_json_dict(json_dict)

    @override
    def close(self) -> None:
        """Close the exporter.

        Implementation of Exporter.close() for exporters that export the data as JSON.
        See the docstrings of the Exporter interface for more details.
        """
        if self._last_start_span:
            self._record_span(self._last_start_span, RecordType.START)
            _logger.warning(f"Span {self._last_start_span.id} was not closed")
            self._last_start_span = None
        super().close()

    def _record_span(self, span: Span, record_type: RecordType) -> None:
        """Record a span and its attributes as a JSON dictionary.

        Args:
            span: The span to record.
            record_type: The type of record to record.
        """
        json_dict: Dict[str, Any] = {
            "type": record_type.value,
            "id": str(span.id),
            "name": span.name_str,
        }
        if span.attributes:
            json_dict["attributes"] = span.attributes.to_json()

        if record_type == RecordType.START:
            json_dict["start_event"] = self._event_data_to_json(span.start_event)
        elif record_type == RecordType.COMPLETE:
            assert_that(span.stop_event, "cannot export a complete record type for an active span!")
            json_dict["start_event"] = self._event_data_to_json(span.start_event)
            json_dict["stop_event"] = self._event_data_to_json(span.stop_event)  # type: ignore[union-attr]
        elif record_type == RecordType.STOP:
            assert_that(span.stop_event, "missing stop event!")
            json_dict["stop_event"] = self._event_data_to_json(span.stop_event)  # type: ignore[union-attr]
        else:
            raise OneLoggerError(f"Invalid type: {record_type}")

        self._export_json_dict(json_dict)

    def _export_json_dict(self, json_dict: Dict[str, Any]) -> None:
        """Export a JSON-compatible dictionary after adding a few extra entries.

        Args:
            json_dict: The JSON-compatible dictionary to write.
        """
        with self._lock:
            try:
                self._increment_record_count()
                json_dict["count"] = self._record_count
                json_dict["pid"] = self._pid
                self._write(json_dict)
            except Exception as e:
                _logger.error(f"Error exporting json dict: {e}")
                raise e

    @abstractmethod
    def _write(self, json_dict: Dict[str, Any]) -> None:
        """Write a JSON-compatible dictionary to the output as a new record.

        Args:
            json_dict: The JSON-compatible dictionary to write.
        """
        raise NotImplementedError("Subclasses must implement this method!")

    def _increment_record_count(self) -> int:
        with self._lock:
            self._record_count += 1
            return self._record_count

    def _event_data_to_json(self, event: Event) -> Dict[str, Any]:
        """Convert the exportable data of an event to a JSON-compatible dictionary.

        Event.to_json is meant for general serialization of the event. This method, on the other hand,
        focuses on what we want to export about an event.
        """
        return {
            "name": event.name_str,
            "attributes": event.attributes.to_json(),
        }
