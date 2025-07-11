# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.core.internal.singleton import SingletonMeta
from nv_one_logger.core.span import SpanName
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.recorder.default_recorder import ExportCustomizationMode

from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder

_logger = get_logger(__name__)

# The following spans are not exported by defaul as they are often short and
# occur frequently. Therefore, exporting them would result in a lot of data.
DEFAULT_SPANS_EXPORT_BLACKLIST: List[SpanName] = [
    StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION,
    StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION,
    StandardTrainingJobSpanName.TESTING_SINGLE_ITERATION,
    StandardTrainingJobSpanName.DATA_LOADING,
    StandardTrainingJobSpanName.MODEL_FORWARD,
    StandardTrainingJobSpanName.ZERO_GRAD,
    StandardTrainingJobSpanName.MODEL_BACKWARD,
    StandardTrainingJobSpanName.OPTIMIZER_UPDATE,
]


class TrainingTelemetryProvider(metaclass=SingletonMeta["TrainingTelemetryProvider"]):
    """A singleton class provider for training telemetry.

    This singleton provides a global point of entry for the training telemetry library. Note that this singleton is a wrapper around
    the OneLoggerProvider singleton.
    The main purpose of this singleton is:
    - Ensuring we have a single Recorder instance per process and making that instance
      available everywhere (see the comments on OneLoggerProvider).
    - Supports building the TrainingTelemetryConfig incrementally by calling with_xxx methods (the builder pattern).
    - Ensures that the config passed to OneLoggerProvider.instance().configure() is a
      TrainingTelemetryConfig and the recorder is a TrainingRecorder so that the training telemetry library can be used
      without having to worry about the config and recorder being passed correctly or downcasting them on each access.

    This singleton needs to be configured once per process as follows on application start up:
    Example usage:
    ```python
        config = TrainingTelemetryConfig(....) # You can use a factory that takes a json or other representation of
        the configs and creates a TrainingTelemetryConfig object.
        TrainingTelemetryProvider.instance().configure(config=config, exporters=[...])
    ```

    Once configured, the application can use the singleton:
    ```python
        # In other parts of the application
        with TrainingTelemetryProvider.instance().recorder.start("my_span"):
            ... # code here will be considered part of the span and will be timed/recorded.

        or use timed_span context manager.
    ```

    Note that the following assumptions are valid:
    - OneLoggerProvider.instance().config() == TrainingTelemetryProvider.instance().config()
    - OneLoggerProvider.instance().recorder() == TrainingTelemetryProvider.instance().recorder()
    - All methods of OneLoggerProvider.instance() are available on TrainingTelemetryProvider.instance() and return the exact same value.

    ```python
    # Example 1: Use a TrainingTelemetryConfig object to configure the training telemetry.
    config = TrainingTelemetryConfig(
        world_size_or_fn=lambda: 4,
        global_batch_size_or_fn=lambda: 32,
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_session",
        ...)
    exporter = LoggerExporter(..)
    (TrainingTelemetryProvider.instance()
        .with_base_telemetry_config(config)
        .with_exporter(exporter)
        .configure_provider())


    # Example 2:  Use a TrainingTelemetryConfig object as base config but override individual fields.
    config = TrainingTelemetryConfig(...)
    exporter = LoggerExporter(..)
    (TrainingTelemetryProvider.instance()
        .with_base_telemetry_config(config)
        .with_config_override(
            {
                "world_size_or_fn": 8,
                "global_batch_size_or_fn": 64,
                "application_name": "test_app",
            }
        )
        .with_exporter(exporter)
        .configure_provider())

    # Example 3:  Build the config by incrementally adding fields (and possibly overriding previous fields).
    exporter = LoggerExporter(..)
    (TrainingTelemetryProvider.instance()
        .with_config_override(
            {
                "world_size_or_fn": 8,
                "global_batch_size_or_fn": 64,
                "application_name": "test_app",
            }
        .with_config_override(
            {
                "world_size_or_fn": 4,
            }
        .with_config_override(
            {
                ....
            }
        )
        .with_exporter(exporter)
        .configure_provider())

    # Example 4: Add multiple exporters.
    exporter1 = LoggerExporter(..)
    exporter2 = OtelExporter(..)
    (TrainingTelemetryProvider.instance()
        .with_base_telemetry_config(config)
        .with_exporter(exporter1)
        .with_exporter(exporter2)
        .configure_provider())


    # Example 5: Set the export customization mode.
    exporter1 = LoggerExporter(..)
    exporter2 = OtelExporter(..)
    (TrainingTelemetryProvider.instance()
        .with_base_telemetry_config(config)
        .with_exporter(exporter1)
        .with_exporter(exporter2)
        .with_export_customization(export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter=[...])
        .configure_provider())

    ```

    """

    def __init__(self):
        # Tentative/temporary internal state built incrementally by the calling with_xxx methods.
        self.__tmp_base_config: Optional[TrainingTelemetryConfig] = None
        self.__tmp_config_overrides: Dict[str, Any] = {}
        self.__tmp_exporters: List[Exporter] = []
        self.__tmp_export_customization_mode: Optional[ExportCustomizationMode] = None
        self.__tmp_span_name_filter: Optional[List[SpanName]] = None

        self.__fully_configured: bool = False

    def with_base_telemetry_config(self, telemetry_config: TrainingTelemetryConfig) -> "TrainingTelemetryProvider":
        """Set the base config for the training telemetry.

        Subsequent calls to "with_config_override_xxx" will override fields of the base config.

        Note: You can only call this method once.
        """
        if self.__fully_configured:
            raise OneLoggerError("with_base_telemetry_config can be called only before configure_provider is called.")
        if self.__tmp_base_config is not None:
            raise OneLoggerError("You can only call with_base_telemetry_config once")
        self.__tmp_base_config = telemetry_config
        return self

    def with_config_override(self, partial_config: Dict[str, Any]) -> "TrainingTelemetryProvider":
        """Add a partial or full config to the training telemetry.

        This will override fields of the base config as well as fields set by
        previous calls to "with_config_override_xxx".

        Note: You can call this method multiple times to add multiple partial configs.
        """
        if self.__fully_configured:
            raise OneLoggerError("with_config_override can be called only before configure_provider is called.")
        self.__tmp_config_overrides.update(partial_config)
        return self

    def with_config_override_from_json_file(self, file_path: Path) -> "TrainingTelemetryProvider":
        """Add a partial or full config to the training telemetry from a JSON file.

        This will override fields of the base config as well as fields set by
        previous calls to "with_config_override_xxx".

        Note: You can call this method multiple times to add multiple partial configs.
        """
        if self.__fully_configured:
            raise OneLoggerError("with_config_override_from_json_file can be called only before configure_provider is called.")
        raise NotImplementedError("Not implemented")

    def with_config_override_from_config_server(self, config_server_url: str) -> "TrainingTelemetryProvider":
        """Add a partial or full config to the training telemetry from a config server.

        This will override fields of the base config as well as fields set by
        previous calls to "with_config_override_xxx".

        Note: You can call this method multiple times to add multiple partial configs.
        """
        if self.__fully_configured:
            raise OneLoggerError("with_config_override_from_config_server can be called only before configure_provider is called.")
        raise NotImplementedError("Not implemented")

    def with_exporter(self, exporter: Exporter) -> "TrainingTelemetryProvider":
        """Add an exporter to the training telemetry.

        Note: You can call this method multiple times to add multiple exporters.
        """
        if self.__fully_configured:
            raise OneLoggerError("with_exporter can be called only before configure_provider is called.")
        self.__tmp_exporters.append(exporter)
        return self

    def with_export_customization(self, export_customization_mode: ExportCustomizationMode, span_name_filter: List[SpanName]) -> "TrainingTelemetryProvider":
        """Set the export customization mode and span name filter for the training telemetry.

        Note: This method is optional but can be called at most once.
              If not called, the default export customization mode will be used.

        export_customization_mode: The mode of exporting spans (and their associated events and attributes) to exporters.
            By default, we export all spans except the ones in the DEFAULT_SPANS_EXPORT_BLACKLIST blacklist.
        span_name_filter: This argument should be interpretted wrt the value of export_customization_mode:
            If export_customization_mode is ExportCustomizationMode.EXPORT_ALL_SPANS, span_name_filter should not be set.
            If export_customization_mode is ExportCustomizationMode.WHITELIST_SPANS, span_name_filter is a list of span names to export (whitelist).
            If export_customization_mode is ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter is a list of span names to not export (blacklist).
            By default, we export all spans except the ones in the DEFAULT_SPANS_EXPORT_BLACKLIST blacklist.
        """
        if self.__fully_configured:
            raise OneLoggerError("with_export_customization can be called only before configure_provider is called.")
        if self.__tmp_export_customization_mode is not None or self.__tmp_span_name_filter is not None:
            raise OneLoggerError("You can only call with_export_customization once")

        self.__tmp_export_customization_mode = export_customization_mode
        self.__tmp_span_name_filter = span_name_filter
        return self

    def configure_provider(self) -> None:
        """Use the providedconfig and exporters to make the training telemetry provider ready to use.

        You can safely use callbacks, context managers,
        or access the TrainingRecorder through TrainingTelemetryProvider.instance().recorder after this call.

        Note: You can only call this method once per application.
        """
        if self.__fully_configured:
            raise OneLoggerError("You can only call configure_provider once per application.")

        config = self._build_telemetry_config()
        export_customization_mode = ExportCustomizationMode.BLACKLIST_SPANS
        if self.__tmp_export_customization_mode is not None:
            export_customization_mode = self.__tmp_export_customization_mode

        span_name_filter = DEFAULT_SPANS_EXPORT_BLACKLIST
        if self.__tmp_span_name_filter is not None:
            span_name_filter = self.__tmp_span_name_filter

        if not config.enable_for_current_rank:
            # Exporters can have non-trivial initialization logic (e.g., connecting to a remote server)
            # that we don't need to run if the training telemetry is disabled.
            exporters = []
        else:
            exporters = self.__tmp_exporters

        if not exporters:
            _logger.warning("No exporters were provided. This means that no telemetry data will be collected.")

        recorder = TrainingRecorder(config=config, exporters=exporters, export_customization_mode=export_customization_mode, span_name_filter=span_name_filter)
        OneLoggerProvider.instance().configure(config, recorder)
        self.__fully_configured = True

        if not config.enable_for_current_rank:
            OneLoggerProvider.instance().force_disable_logging()

    def _build_telemetry_config(self) -> TrainingTelemetryConfig:
        if self.__tmp_base_config is None and not self.__tmp_config_overrides:
            raise OneLoggerError("No configuration was provided. Please provide a base config and/or config overrides.")
        merged_config = self.__tmp_base_config.model_dump() if self.__tmp_base_config else {}
        merged_config.update(self.__tmp_config_overrides)
        try:
            return TrainingTelemetryConfig(**merged_config)
        except Exception as e:
            raise OneLoggerError(f"Invalid configuration! Did you forget some required fields? : {e}") from e

    @property
    def recorder(self) -> TrainingRecorder:
        """Return the recorder."""
        recorder = OneLoggerProvider.instance().recorder
        assert_that(
            self.__fully_configured,
            "You need to call TrainingTelemetryProvider.instance().configure() once in your application before accessing the recorder.",
        )
        assert_that(
            recorder and isinstance(recorder, TrainingRecorder),
            "The recorder returned by TrainingTelemetryProvider.instance().recorder is not a TrainingRecorder.",
        )
        return cast(TrainingRecorder, recorder)

    @property
    def config(self) -> TrainingTelemetryConfig:
        """Return the config."""
        config = OneLoggerProvider.instance().config
        assert_that(
            self.__fully_configured,
            "You need to call TrainingTelemetryProvider.instance().configure() once in your application before accessing the recorder.",
        )
        assert_that(
            config and isinstance(config, TrainingTelemetryConfig),
            f"The config returned by TrainingTelemetryProvider.instance().config is not a TrainingTelemetryConfig but was of type {type(config)}.",
        )
        return cast(TrainingTelemetryConfig, config)

    @property
    def one_logger_ready(self) -> bool:
        """Check if the one_logger is ready to be used."""
        assert_that(
            self.__fully_configured == OneLoggerProvider.instance().one_logger_ready,
            "Internal inconsistency: The one_logger_ready property of TrainingTelemetryProvider.instance() and OneLoggerProvider.instance() are not in sync.",
        )
        return OneLoggerProvider.instance().one_logger_ready

    def force_disable_logging(self) -> None:
        """Force logging to be disabled effectively disabling onelogger library."""
        OneLoggerProvider.instance().force_disable_logging()
