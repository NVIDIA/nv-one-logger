# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
import multiprocessing.connection
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import wandb
from nv_one_logger.core.attributes import Attribute
from nv_one_logger.core.event import ErrorEvent, Event, StandardEventName, TelemetryDataError
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.utils import temporarily_modify_env
from nv_one_logger.core.span import Span
from nv_one_logger.exporter.exporter import BaseExporter, ExportError
from overrides import EnforceOverrides, override
from wandb.sdk.wandb_run import Run


@dataclass
class Config:
    """Configuration for the WandBExporter."""

    # An entity is a username or team name where you're sending runs. This entity must exist before you can send runs.
    # See wandb.init() for more details.
    entity: str

    # The URL of the WandB server.
    host: str = "https://api.wandb.ai"

    # The API authentication key for the WandB server.
    api_key: str = ""

    # The W*B project name to use for storing data.
    project: str = "one-logger-tracking"

    # A short display name of the W&B run. All the exported data will be associated with this run.
    run_name: str = f"one-logger-tracking-run-{str(uuid.uuid4())}"

    # An absolute path to a directory where metadata will be stored.
    # See the "dir" param ofwandb.init() for more details.
    save_dir: str = "./wandb"

    # A list of strings, which will populate the list of tags on this run in the W&B UI.
    # Tags are useful for organizing runs together, or applying temporary labels like "baseline" or "production".
    # See wandb.init() for more details.
    tags: List[str] = field(default_factory=lambda: ["e2e_metrics_enabled"])


# Keep this function out of the class. Calling object methods across multiple processes is not a good idea
# as you would work with 2 copies of the object!
def _initialize_wandb(config: Config) -> Run:
    # Authenticate and start a new wandb run for data logging"""
    # NOTE: If wandb init is called from a process that contains WANDB_SERVICE,
    #       it does not create a new wandb service process. This induces interference
    #       between the onelogger’s wandb logging and the customer code’s wandb logging.
    #       So we need to temporarily modify the environment variables.
    with temporarily_modify_env(_ENV_VAR_WANDB_RUN_ID), temporarily_modify_env(_ENV_VAR_WANDB_SERVICE):
        wandb.login(anonymous="allow", key=config.api_key, host=config.host, timeout=5)
        return wandb.init(
            project=config.project,
            name=config.run_name,
            entity=config.entity,
            tags=config.tags,
            dir=config.save_dir,
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore[assignment]


class MetricNamingStrategy(ABC, EnforceOverrides):
    """An interface for determining how to generate metric names for span and event attributes.

    Given that W&B doesn't have the concept of spans or traces (see the docstrings for WandBExporterBase), we need
    to somehow map each attribute to a metric. Users can use one of the canned implementations or create their own implementation
    to control the metric naming strategy.
    """

    @abstractmethod
    def from_span_attribute(self, span: Span, attribute: Attribute) -> str:
        """Generate a metric name from a span attribute.

        :param span: The span that the attribute belongs to.
        :param attribute: The attribute to generate a metric name from.
        :return: A metric name.
        """
        raise NotImplementedError

    @abstractmethod
    def from_event_attribute(self, span: Span, event: Event, attribute: Attribute) -> str:
        """Generate a metric name from an event attribute.

        :param span: The span that the event belongs to.
        :param event: The event that the attribute belongs to.
        :param attribute: The attribute to generate a metric name from.
        :return: A metric name.
        """
        raise NotImplementedError


class HierarchicalMetricNamingStrategy(MetricNamingStrategy):
    """A strategy that generates a metric name by joining the span name, event name, and attribute name with a separator.

    This is a simple and intuitive metric naming strategy that is easy to understand and use.
    - For span attributes, the metric name is created by concatenating the span name and the attribute name.
    - For event attributes, the metric name is created by concatenating the span name, the event name, and the attribute name.


    For example,
        an attribute named" foo" for a span named "my_span" will be reported as a metric named "my_span.foo" and
        an attribute named "bar" for an event named "my_event" of span "my_span" will be reported as a metric named "my_span.my_event.bar".
    """

    def __init__(self, separator: str = "."):
        """
        Initialize the HierarchicalMetricNamingStrategy.

        :param separator: The separator to use between the span name, event name, and attribute name. Default is ".".
        """
        self._separator = separator

    @override
    def from_span_attribute(self, span: Span, attribute: Attribute) -> str:
        """Generate a metric name from a span attribute by joining the span name and the attribute name with the separator."""
        return self._separator.join([span.name_str, attribute.name])

    @override
    def from_event_attribute(self, span: Span, event: Event, attribute: Attribute) -> str:
        """Generate a metric name from an event attribute by joining the span name, event name, and the attribute name with the separator."""
        return self._separator.join([span.name_str, event.name_str, attribute.name])


class FlatMetricNamingStrategy(MetricNamingStrategy):
    """A strategy that generates a metric name solely based on the attribute name.

    This implementation is useful when users want to export all attributes as metrics of the same name. It simply uses the
    name of the attribute as the metric name.

    For example,
        an attribute named" foo" for a span named "my_span" will be reported as a metric named "foo" and
        an attribute named "bar" for an event named "my_event" of span "my_span" will be reported as a metric named "bar".
    """

    @override
    def from_span_attribute(self, span: Span, attribute: Attribute) -> str:
        """Generate a metric name from a span attribute by using the attribute name."""
        return attribute.name

    @override
    def from_event_attribute(self, span: Span, event: Event, attribute: Attribute) -> str:
        """Generate a metric name from an event attribute by using the attribute name."""
        return attribute.name


_ENV_VAR_WANDB_RUN_ID = "WANDB_RUN_ID"
_ENV_VAR_WANDB_SERVICE = "WANDB_SERVICE"
_DEFAULT_METRIC_NAMING_STRATEGY = HierarchicalMetricNamingStrategy()


def get_current_time_msec() -> int:
    """Return current wall-clock time in milliseconds since epoch."""
    return round(time.time() * 1000.0)


class WandBExporterBase(BaseExporter):
    """Base class for Exporter implementations that send spans and events to Weights and Biases.

    NOTE:
        Weights and Biases doesn't have the concept of spans or traces as it is not designed as a tracing system. It
        can store only "metrics". Therefore, this exporter simply exports attributes values from spans and events as metrics
        and sends them to wandb.
    """

    def __init__(self, config: Config, metric_naming_strategy: MetricNamingStrategy) -> None:
        """
        Initialize the WandBExporterBase.

        :param config: The configuration for the WandBExporter.
        :param metric_naming_strategy: The strategy to use for naming metrics based on span and event names.
        """
        super().__init__()

        assert_that(config, "config is required.")
        assert_that(metric_naming_strategy, "metric_naming_strategy is required.")
        self._config = config
        self._metric_naming_strategy = metric_naming_strategy

        self._exit_code = 0
        self._run: Optional[Run] = None  # Set only after initialize() is called.

    @override
    def initialize(self) -> None:
        """Initialize the WandBExporterBase."""
        super().initialize()

    @override
    def export_start(self, span: Span) -> None:
        """Export the start of a span.

        This implementation exports the attributes of the span and the start event as metrics.
        See the docstring of the Exporter class for more details.
        """
        super().export_start(span)
        metrics = {self._metric_naming_strategy.from_span_attribute(span, attrib): attrib.value for attrib in span.attributes.values()}
        start_event_attributes = {
            self._metric_naming_strategy.from_event_attribute(span, span.start_event, attrib): attrib.value for attrib in span.start_event.attributes.values()
        }
        metrics.update(start_event_attributes)
        self._log_metrics(metrics)

    @override
    def export_stop(self, span: Span) -> None:
        """Export the stop of a span.

        This implementation exports the updated attributes of the span and the stop event as metrics.
        See the docstring of the Exporter class for more details.
        """
        super().export_stop(span)

        metrics = {}
        # Attributes set during span cretion are already exported when export_start() was called.
        # So here, we only send span.updated_attributes.
        metrics = {self._metric_naming_strategy.from_span_attribute(span, attrib): attrib.value for attrib in span.updated_attributes.values()}
        assert_that(span.stop_event, "Span has no stop event")
        stop_event_attributes = {
            self._metric_naming_strategy.from_event_attribute(span, span.stop_event, attrib): attrib.value for attrib in span.stop_event.attributes.values()
        }
        metrics.update(stop_event_attributes)
        self._log_metrics(metrics)

    @override
    def export_event(self, event: Event, span: Span) -> None:
        """Export an event.

        This implementation exports the attributes of the event as metrics.
        See the docstring of the Exporter class for more details.
        """
        super().export_event(event, span)
        metrics = {self._metric_naming_strategy.from_event_attribute(span, event, attrib): attrib.value for attrib in event.attributes.values()}
        self._log_metrics(metrics)

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Export an error event.

        This implementation doesn't do anything as wandb doesn't support error reporting.
        """
        super().export_error(event, span)

    @override
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Export a telemetry data error.

        This implementation reports telemetry data errors as a metric with a value that determines
        whether the data is completely missing or is partial/incorrect.

        See the docstring of the Exporter class for more details.
        """
        super().export_telemetry_data_error(error)
        # wandb doesn't have a very expressive data model. So we report telemetry data errors as a metric with
        # a value that determines whether the data is completely missing or is partial/incorrect.
        metrics = {StandardEventName.TELEMETRY_DATA_ERROR.value: error.error_type.name}
        self._log_metrics(metrics)

    @override
    def close(self) -> None:
        """Close the WandBExporterBase."""
        super().close()

    @abstractmethod
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log the elements of the given dictionary as metrics.

        :param metrics: A dictionary of metric names and their values.
        raise ExportError if the metrics cannot be logged.
        """
        raise NotImplementedError


class WandBExporterSync(WandBExporterBase):
    """Exporter implementation that sends spans and events to Weights and Biases synchronously.

    Calls to export_xxx() function  will block until the data is sent to WandB.

    NOTE:
        Weights and Biases doesn't have the concept of spans or traces as it is not designed as a tracing system. It
        can store only "metrics". Therefore, this exporter simply exports attributes values from spans and events as metrics
        and sends them to wandb.
    """

    def __init__(self, config: Config, metric_naming_strategy: MetricNamingStrategy = _DEFAULT_METRIC_NAMING_STRATEGY) -> None:
        """Initialize the WandBExporterSync.

        :param config: The configuration for the WandBExporter.
        :param metric_naming_strategy: The strategy to use for naming metrics based on span and event names. The default is HierarchicalMetricNamingStrategy,
        which generates a metric name by joining the span name, event name, and attribute name with "." as separator, For example,
        an attribute named" foo" for a span named "my_span" will be reported as a metric named "my_span.foo" and
        an attribute named "bar" for an event named "my_event" of span "my_span" will be reported as a metric named "my_span.my_event.bar".
        """
        super().__init__(config=config, metric_naming_strategy=metric_naming_strategy)

    @override
    def initialize(self) -> None:
        """Initialize the WandBExporterSync."""
        super().initialize()
        self._run = _initialize_wandb(self._config)

    @override
    def close(self) -> None:
        """Close the WandBExporterSync."""
        # Mark a run as finished, and finish uploading all data.
        if self._run:
            self._run.finish(exit_code=self._exit_code, quiet=False)
            self._run = None
        super().close()

    @override
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        if not metrics:
            return
        metrics["app_last_log_time"] = get_current_time_msec()
        self._run.log(data=metrics, commit=True)


@dataclass
class _WandBExportAsyncErrorInfo:
    """Information about an error that occurred during asynchronous WandB export."""

    message: str
    traceback: str


class WandBExporterAsync(WandBExporterBase):
    """Exporter implementation that sends spans and events to Weights and Biases asynchronously in a separate child process..

    This class uses a child process to send data to WandB. This avoids any potential conflicts with the
    wandb instance in user's application (if the user is using wandb for purposes others than onelogger).

    NOTE:
        Weights and Biases doesn't have the concept of spans or traces as it is not designed as a tracing system. It
        can store only "metrics". Therefore, this exporter simply exports attributes values from spans and events as metrics
        and sends them to wandb.
    """

    def __init__(self, config: Config, metric_naming_strategy: MetricNamingStrategy = _DEFAULT_METRIC_NAMING_STRATEGY) -> None:
        """Initialize the WandBExporterAsync.

        :param config: The configuration for the WandBExporter.
        :param metric_naming_strategy: The strategy to use for naming metrics based on span and event names. The default is HierarchicalMetricNamingStrategy,
        which generates a metric name by joining the span name, event name, and attribute name with "." as separator, For example,
        an attribute named" foo" for a span named "my_span" will be reported as a metric named "my_span.foo" and
        an attribute named "bar" for an event named "my_event" of span "my_span" will be reported as a metric named "my_span.my_event.bar".
        """
        super().__init__(config=config, metric_naming_strategy=metric_naming_strategy)

        # Create a duplex pipe for bidirectional communication between parent and child processes
        # We need duplex=True because:
        # 1. Parent sends metrics data to child (_parent_conn)
        # 2. Child sends error information back to parent (_child_conn)
        conn_to_parent, self._conn_to_child = mp.Pipe(duplex=True)
        self._process = mp.Process(target=WandBExporterAsync._child_process_fn, args=[conn_to_parent, self._config], daemon=True)
        self._first_metrics_call = True  # Flag to track first call to log_metrics

    @override
    def initialize(self) -> None:
        """Initialize the WandBExporterAsync."""
        self._process.start()
        super().initialize()
        # Do not call initialize wandb yet! It will be called in the child process.

    @staticmethod  # Keep this method static. Calling object methods across multiple processes is not a good idea.
    def _child_process_fn(conn_to_parent: multiprocessing.connection.Connection, config: Config) -> None:  # noqa: C901
        """Receive the data sent to the child process and log to WandB."""
        wandb_run = None
        error_info = None
        try:
            # Initialize outside the main loop to avoid repeated initializations.
            wandb_run = _initialize_wandb(config)
            assert wandb_run is not None, "WandB run was not initialized in the child process."

            # Main processing loop - continue until explicitly told to exit.
            while True:  # type: ignore[unreachable]
                # Get data from the pipe
                data = conn_to_parent.recv()
                if data is None:
                    break  # Exit the loop ONLY if None is explicitly received as a signal to exit (without an error).
                wandb_run.log(data=data, commit=True)  # type: ignore[attr-defined]
        except EOFError:
            # This can happen if the parent process closes the connection.
            pass
        except Exception as e:
            # Send the error to the parent process
            error_info = _WandBExportAsyncErrorInfo(message=f"Unexpected error in child process: {str(e)}", traceback=traceback.format_exc())
            try:
                conn_to_parent.send(error_info)
            except Exception:
                pass  # If we can't send through pipe, we can't do much

        # Clean shutdown of the child process when explicitly told to exit.
        exit_code = 1 if error_info else 0
        try:
            if wandb_run:
                wandb_run.finish(exit_code=exit_code, quiet=True)  # type: ignore[unreachable]
                wandb_run = None
        except Exception:
            pass  # If wandb finish fails, we've tried our best and can't do muich more.
        finally:
            conn_to_parent.close()
            # This ensures the process.exitcode seen by the parent process will be correct.
            # https://docs.python.org/3.11/library/multiprocessing.html#multiprocessing.Process.exitcode
            if exit_code != 0:
                sys.exit(exit_code)

    @override
    def close(self) -> None:
        """Close the WandBExporterAsync."""
        try:
            self._conn_to_child.send(None)

            # Wait a moment to give the child time to process the shutdown signal
            self._conn_to_child.poll(timeout=0.1)

            # Check for any errors that might have occurred during shutdown
            self._check_for_errors()
        except Exception:
            pass
        # Wait for the process to completely terminate
        self._process.join()

        super().close()

    def _check_for_errors(self) -> None:
        """
        Check if there are any error messages from the child process.

        This method polls the connection to see if there's any data available.
        If an error is found, it raises a RuntimeError.

        This is called at the beginning of every public API method to ensure
        errors are detected as soon as possible.
        """
        # First check if the child process has terminated unexpectedly.
        # The hasattr check protects against race conditions where a process is detected as not alive
        # but the exitcode attribute hasn't been set yet by the multiprocessing system.
        if not self._process.is_alive() and hasattr(self._process, "exitcode") and self._process.exitcode != 0:
            # Child process has terminated with a non-zero exit code
            error_msg = f"WandB child process terminated unexpectedly with exit code {self._process.exitcode}"
            raise ExportError(error_msg)

        try:
            # Check if there's any data in the pipe, but don't block
            if self._conn_to_child.poll(0):  # 0 timeout means non-blocking check
                data = self._conn_to_child.recv()
                # If the received data is an error message
                if isinstance(data, _WandBExportAsyncErrorInfo):
                    raise ExportError(f"WandB logging failed: {data.message}\n{data.traceback}")
        # Specifically catching pipe communication errors:
        # - EOFError: Raised when pipe is closed on the other end
        # - OSError/IOError: General I/O error during pipe operations (includes BrokenPipeError which occurs when writing to a closed pipe)
        except (EOFError, OSError) as pipe_error:
            # Pipe broken or closed - just store the error info, don't try to send it
            # (we're in the parent process detecting the error, not reporting it)
            if self._process.is_alive():
                # Pipe error while process is still running is unexpected
                error_msg = f"Pipe communication error while child process is still running: {str(pipe_error)}"
            else:
                # Child process has exited, which explains the pipe error
                error_msg = f"Child process has terminated, causing pipe error: {str(pipe_error)}"

            raise OneLoggerError(f"WandB logging failed: {error_msg}")

    @override
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        if not metrics:
            return

        # Check for any errors before attempting to log metrics
        self._check_for_errors()

        assert_that(self.ready, "Exporter is not ready. Cannot log metrics.")
        # Send metrics data to the child process
        metrics["app_last_log_time"] = get_current_time_msec()
        self._conn_to_child.send(obj=metrics)

        # On the first call to wandb, do a check to make sure everything worked as expected.
        if self._first_metrics_call:
            # Wait for possible error messages from child process with timeout
            # This is more responsive than sleep as it will return immediately if data is available
            self._conn_to_child.poll(timeout=0.1)

            # Check for errors again after sending metrics
            # This will catch any immediate errors triggered by processing the metrics
            self._check_for_errors()

            # Mark that we've done the first call check
            self._first_metrics_call = False
