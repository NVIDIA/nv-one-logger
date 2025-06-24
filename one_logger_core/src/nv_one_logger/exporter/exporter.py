# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from enum import Enum, auto

from overrides import EnforceOverrides, override

from nv_one_logger.core.event import ErrorEvent, Event, TelemetryDataError
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.span import Span


class ExportError(OneLoggerError):
    """Error raised when exporting data fails."""

    def __init__(self, message: str):
        super().__init__(message)


class Exporter(ABC, EnforceOverrides):
    """Interface for exporting data from the instrumented application to an observability backend or a collector pipeline."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the exporter.

        An exporter can be used only after being initialized.

        Raises:
            ExportError: If there is an error during the initialization operation.
        """

    @abstractmethod
    def export_start(self, span: Span) -> None:
        """Export a newly started span along with its current attributes and its start event.

        Args:
            span: The span to export. This span can be still in progress.

        Raises:
            ExportError: If there is an error during the export operation.
        """

    @abstractmethod
    def export_stop(self, span: Span) -> None:
        """Export a stopped/finished span along with its current attributes and its stop event.

        Args:
            span: The span to export. This span must be already stopped.

           This span must be already stopped.

        Raises:
            ExportError: If there is an error during the export operation.
        """

    @abstractmethod
    def export_event(self, event: Event, span: Span) -> None:
        """Export an event that occurred for an active span.

        Note: you do not need to call this method for start and stop events as they are exported in the export_start and export_stop methods.
        Args:
            event: The event to export.
            span: The span that the event belongs to. This span must be still active.

        Raises:
            ExportError: If there is an error during the export operation.
        """

    @abstractmethod
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Export an error event that occurred for an active span.

        Args:
            event: The error event to export.
            span: The span that the error event belongs to. This span must be still active.

        Raises:
            ExportError: If there is an error during the export operation.
        """

    @abstractmethod
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Export a telemetry data error.

        This method is used to signal to the telemetry backend that the telemetry data is not complete/correct.

        Args:
            issue: The telemetry data issue to export.

        Raises:
            ExportError: If there is an error during the export operation.
        """

    @abstractmethod
    def close(self) -> None:
        """Shut down the exporter.

        Use this method to release any resources held by the exporter or flush any pending data that is not yet exported.
        Raises:
            ExportError: If there is an error during the export operation.
        """


class BaseExporter(Exporter):
    """Base class for all exporters that need to be initialized.

    This base class ensures that the exporter is initialized before it is used and not used after it is closed.
    """

    class _State(Enum):
        """State of the exporter."""

        NOT_INITIALIZED = auto()
        READY = auto()
        CLOSED = auto()

    def __init__(self) -> None:
        self._state: BaseExporter._State = self._State.NOT_INITIALIZED

    @override
    def initialize(self) -> None:
        """Initialize the exporter."""
        self._state = self._State.READY

    @override
    def export_start(self, span: Span) -> None:
        """Export a newly started span along with its current attributes and its start event."""
        self._check_state()

    @override
    def export_stop(self, span: Span) -> None:
        """Export a stopped/finished span along with its current attributes and its stop event."""
        self._check_state()

    @override
    def export_event(self, event: Event, span: Span) -> None:
        """Export an event that occurred for an active span."""
        self._check_state()

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Export an error event that occurred for an active span."""
        self._check_state()

    @override
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Export a telemetry data error."""
        self._check_state()

    @override
    def close(self) -> None:
        """Shut down the exporter."""
        self._state = self._State.CLOSED

    @property
    def ready(self) -> bool:
        """Check if the exporter is in ready state.

        Returns:
            bool: True if the exporter is initialized and ready to use, False otherwise.
        """
        return self._state == self._State.READY

    @property
    def closed(self) -> bool:
        """Check if the exporter is in ready state.

        Returns:
            bool: True if the exporter is initialized and ready to use, False otherwise.
        """
        return self._state == self._State.CLOSED

    def _check_state(self) -> None:
        if self._state == self._State.NOT_INITIALIZED:
            raise ExportError("Exporter not initialized. Call initialize first.")
        if self._state == self._State.CLOSED:
            raise ExportError("Exporter is closed and cannot be used anymore.")
