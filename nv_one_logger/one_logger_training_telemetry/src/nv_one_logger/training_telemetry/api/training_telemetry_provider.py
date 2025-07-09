# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, cast

from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.internal.singleton import SingletonMeta
from nv_one_logger.core.span import SpanName
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.recorder.default_recorder import ExportCustomizationMode

from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder

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
    The main value of this singleton is that it ensures that the config passed on OneLoggerProvider.instance().configure() is a
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

    """

    def configure(
        self,
        config: TrainingTelemetryConfig,
        exporters: List[Exporter],
        export_customization_mode: ExportCustomizationMode = ExportCustomizationMode.BLACKLIST_SPANS,
        span_name_filter: Optional[List[SpanName]] = DEFAULT_SPANS_EXPORT_BLACKLIST,
    ) -> None:
        """
        Set the recorder for the singleton.

        Args:
            config: The configuration for the training telemetry.
            exporters: A list of exporters to use for recording training telemetry.
            export_customization_mode: The mode of exporting spans (and their associated events and attribytes) to exporters.
            export_customization_mode: The mode of exporting spans (and their associated events and attribytes) to exporters.
                By default, we export all spans except the ones in the DEFAULT_SPANS_EXPORT_BLACKLIST blacklist.
            span_name_filter: This argument should be interpretted wrt the value of export_customization_mode:
                If export_customization_mode is ExportCustomizationMode.EXPORT_ALL_SPANS, span_name_filter should not be set.
                If export_customization_mode is ExportCustomizationMode.WHITELIST_SPANS, span_name_filter is a list of span names to export (whitelist).
                If export_customization_mode is ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter is a list of span names to not export (blacklist).
        """
        assert_that(config, "config cannot be None.")
        config.validate_config()
        self._config = config

        if not config.enable_for_current_rank:
            # Exporters can have non-trivial initialization logic (e.g., connecting to a remote server)
            # that we don't need to run if the training telemetry is disabled.
            exporters = []

        recorder = TrainingRecorder(config=config, exporters=exporters, export_customization_mode=export_customization_mode, span_name_filter=span_name_filter)
        OneLoggerProvider.instance().configure(config, recorder)

        if not config.enable_for_current_rank:
            OneLoggerProvider.instance().force_disable_logging()

    @property
    def recorder(self) -> TrainingRecorder:
        """Return the recorder."""
        recorder = OneLoggerProvider.instance().recorder
        assert_that(recorder, "You need to call TrainingTelemetryProvider.instance().configure() once in your application before accessing the recorder.")
        assert_that(isinstance(recorder, TrainingRecorder), "The recorder returned by TrainingTelemetryProvider.instance().recorder is not a TrainingRecorder.")
        return cast(TrainingRecorder, recorder)

    @property
    def config(self) -> TrainingTelemetryConfig:
        """Return the config."""
        config = OneLoggerProvider.instance().config
        assert_that(config, "You need to call TrainingTelemetryProvider.instance().configure() once in your application before accessing the recorder.")
        assert_that(
            isinstance(config, TrainingTelemetryConfig),
            f"The config returned by TrainingTelemetryProvider.instance().config is not a TrainingTelemetryConfig but was of type {type(config)}.",
        )
        return cast(TrainingTelemetryConfig, config)

    @property
    def one_logger_ready(self) -> bool:
        """Check if the one_logger is ready to be used."""
        return OneLoggerProvider.instance().one_logger_ready

    def force_disable_logging(self) -> None:
        """Force logging to be disabled effectively disabling onelogger library."""
        OneLoggerProvider.instance().force_disable_logging()
