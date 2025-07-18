# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false

"""Contains the V1CompatibleWandbExporterAdapter class, which is used to convert v2 telemetry data to v1-compatible metrics format.

The code in this file ensures that the output of the v2 telemetry data is compatible with the v1 metrics format. That is,
it provides compatibility on the output path of v2: It allows using v2 config and using v2 library while producing v1-compatible metrics.
"""

import hashlib
import os
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, cast
import uuid

from nv_one_logger.core.event import ErrorEvent, Event
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.version import get_version
from nv_one_logger.core.span import Span, SpanName, StandardSpanAttributeName, StandardSpanName
from nv_one_logger.training_telemetry.api.attributes import (
    CheckpointSaveSpanAttributes,
    OneLoggerInitializationAttributes,
    SaveCheckpointSuccessEventAttributes,
    SyncCheckpointMetricsUpdateAttributes,
    TrainingLoopAttributes,
    TrainingMetricsUpdateAttributes,
    ValidationMetricsUpdateAttributes,
)
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.wandb.exporter.wandb_exporter import Config as WandBConfig
from nv_one_logger.wandb.exporter.wandb_exporter import (
    HierarchicalMetricNamingStrategy,
    WandBExporterAsync,
    WandBExporterSync,
)
from overrides import override


class _MetricMapping(NamedTuple):
    # The name of the metric in v1.
    v1_name: str
    # The value of the metric or None, if the metric is missing.
    value: Any
    # The coefficient to apply to the value (if the value is not None, a no op otherwise).
    # Use this only when the value is numeric.
    coefficient: Optional[float] = None


def _build_metrics(mappings: List[_MetricMapping]) -> Dict[str, Any]:
    """Builds a dictionary of metrics from a list of metric mappings.

    Args:
        mappings (List[_MetricMapping]): Mapping from v1 metric name to value and coefficient. If the value is None, the metric will be omitted.

    Returns:
        Dict[str, Any]: Dictionary of metrics.
    """
    metrics: Dict[str, Any] = {}
    for mapping in mappings:
        if mapping.value is not None:
            if mapping.coefficient is not None:
                assert_that(isinstance(mapping.value, (int, float)), f"The value of {mapping.v1_name} must be numeric but got {type(mapping.value)}")
                metrics[mapping.v1_name] = float(mapping.value) * mapping.coefficient
            else:
                metrics[mapping.v1_name] = mapping.value
    return metrics


class V1CompatibleWandbExporterAdapter:
    """Adapter class to convert v2 telemetry data to v1-compatible metrics format.

    This allows internal users to switch to one logger v2 without the need to change the downstreamconsumers of the telemetry data
    (e.g., data infra that processes the metrics stored in wandb)
    """

    def __init__(self, training_telemetry_config: TrainingTelemetryConfig) -> None:
        """Initialize the adapter with a training telemetry configuration.

        Args:
            training_telemetry_config: The corresponding v2 configuration.
        """
        self._training_telemetry_config = training_telemetry_config

    def extract_v1_metrics_for_span_start(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.stop_event is None,
            f"the span must be an active span but it is stopped. {span}",
        )
        mapping: dict[SpanName, Callable[[Span], dict[str, Any]]] = {
            StandardSpanName.APPLICATION: self._metrics_for_app_start,
            StandardTrainingJobSpanName.MODEL_INIT: self._metrics_for_model_init_start,
            StandardTrainingJobSpanName.DATA_LOADER_INIT: self._metrics_for_dataloader_init_start,
            StandardTrainingJobSpanName.CHECKPOINT_LOAD: self._metrics_for_load_checkpoint_start,
            StandardTrainingJobSpanName.OPTIMIZER_INIT: self._metrics_for_optimizer_init_start,
            StandardTrainingJobSpanName.TRAINING_LOOP: self._metrics_for_training_loop_start,
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC: self._metrics_for_checkpoint_save_start,
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC: self._metrics_for_checkpoint_save_start,
        }
        if span.name in mapping:
            return mapping[span.name](span)
        return {}

    def extract_v1_metrics_for_span_stop(self, span: Span) -> dict[str, Any]:
        assert_that(span.stop_event, f"the span must be have a stop event. {span}")
        mapping: dict[SpanName, Callable[[Span], dict[str, Any]]] = {
            StandardSpanName.APPLICATION: self._metrics_for_app_stop,
            StandardTrainingJobSpanName.MODEL_INIT: self._metrics_for_model_init_stop,
            StandardTrainingJobSpanName.DATA_LOADER_INIT: self._metrics_for_dataloader_init_stop,
            StandardTrainingJobSpanName.CHECKPOINT_LOAD: self._metrics_for_load_checkpoint_stop,
            StandardTrainingJobSpanName.OPTIMIZER_INIT: self._metrics_for_optimizer_init_stop,
            StandardTrainingJobSpanName.TRAINING_LOOP: self._metrics_for_training_loop_stop,
            # We don't need to do anything for CHECKPOINT_SAVE_SYNC and CHECKPOINT_SAVE_ASYNC spans
            # because all the metrics we care about are available as span attributes at the span start time or
            # attributes for the SAVE_CHECKPOINT_SUCCESS and SYNC_CHECKPOINT_METRICS_UPDATE events.
        }
        if span.name in mapping:
            return mapping[span.name](span)
        return {}

    def extract_v1_metrics_for_event(self, event: Event, span: Span) -> dict[str, Any]:
        mapping: dict[StandardTrainingJobEventName, Callable[[Event, Span], dict[str, Any]]] = {
            StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION: self._metrics_for_one_logger_initilzation_event,
            StandardTrainingJobEventName.TRAINING_METRICS_UPDATE: self._metrics_for_training_metrics_update_event,
            StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE: self._metrics_for_validation_metrics_update_event,
            StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE: self._metrics_for_sync_checkpoint_metrics_update_event,
            StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS: self._metrics_for_save_checkpoint_success_event,
        }
        if event.name in mapping:
            return mapping[event.name](event, span)
        return {}

    def _metrics_for_one_logger_initilzation_event(self, event: Event, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardSpanName.APPLICATION,
            "Expected span name to be APPLICATION",
        )
        assert_that(
            event.name == StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION,
            "Expected event name to be ONE_LOGGER_INITIALIZATION",
        )
        assert_that(
            isinstance(event.attributes, OneLoggerInitializationAttributes),
            f"Expected event attributes to be of type 'OneLoggerInitializationAttributes' but got {type(event.attributes)}",
        )
        attributes = cast(OneLoggerInitializationAttributes, event.attributes)
        metrics_to_log: dict[str, Any] = _build_metrics(
            [
                _MetricMapping("summary_data_schema_version", attributes.summary_data_schema_version),
                _MetricMapping("app_run_type", attributes.app_type),
                _MetricMapping("app_metrics_feature_tags", "full"),
                _MetricMapping("is_baseline_run", attributes.is_baseline_run),
                _MetricMapping("is_train_iterations_enabled", attributes.is_train_iterations_enabled),
                _MetricMapping("is_validation_iterations_enabled", attributes.is_validation_iterations_enabled),
                _MetricMapping("is_test_iterations_enabled", attributes.is_test_iterations_enabled),
                _MetricMapping("is_save_checkpoint_enabled", attributes.is_save_checkpoint_enabled),
                _MetricMapping("is_log_throughput_enabled", attributes.is_log_throughput_enabled),
                _MetricMapping("world_size", attributes.world_size),
                _MetricMapping("global_batch_size", attributes.global_batch_size),
                _MetricMapping("app_tag_run_name", attributes.session_tag),
                _MetricMapping("micro_batch_size", attributes.micro_batch_size),
                _MetricMapping("model_seq_length", attributes.seq_length),
                _MetricMapping("save_checkpoint_strategy", attributes.checkpoint_strategy if attributes.is_save_checkpoint_enabled else None),
            ]
        )
        metrics_to_log.update(self._perf_tag_dict(self._training_telemetry_config))

        # We deprecated "app_tag_run_version" for v2, but during the transition, we pass the "app_tag_run_version"
        # value provided by v1 users as custom metadata. See ConfigAdapter.convert_to_v2_config for more details.
        if attributes.custom_metadata:
            reconstructed_dict = {item.split(":")[0]: item.split(":")[1] for item in attributes.custom_metadata}
            overlap_keys = set(metrics_to_log.keys()).intersection(reconstructed_dict.keys())
            if overlap_keys:
                raise OneLoggerError(f"Metadata overlap found with keys: {overlap_keys}")
            metrics_to_log.update(reconstructed_dict)

        return metrics_to_log

    def _perf_tag_dict(self, config: TrainingTelemetryConfig) -> dict[str, Any]:
        perf_tag_list: list[str] = []
        if type(config.perf_tag) is list:
            perf_tag_list = config.perf_tag

        else:
            perf_tag_list = [config.perf_tag]

        perf_tag_id_list = [hashlib.md5(pt.encode("utf-8")).hexdigest() for pt in perf_tag_list]
        return {
            "app_tag": perf_tag_list,
            "app_tag_id": perf_tag_id_list,
            "app_tag_count": len(perf_tag_list),
        }

    def _metrics_for_model_init_start(self, span: Span) -> dict[str, Any]:
        metrics_to_log: dict[str, Any] = _build_metrics([_MetricMapping("app_model_init_start_time", span.start_event.timestamp.milliseconds_since_epoch)])

        return metrics_to_log

    def _metrics_for_model_init_stop(self, span: Span) -> dict[str, Any]:
        return _build_metrics([_MetricMapping("app_model_init_finish_time", span.stop_event.timestamp.milliseconds_since_epoch)])

    def _metrics_for_dataloader_init_start(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.DATA_LOADER_INIT,
            "Expected span name to be DATA_LOADER_INIT",
        )
        metrics_to_log = _build_metrics([_MetricMapping("app_build_dataiters_start_time", span.start_event.timestamp.milliseconds_since_epoch)])
        return metrics_to_log

    def _metrics_for_dataloader_init_stop(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.DATA_LOADER_INIT,
            "Expected span name to be DATA_LOADER_INIT",
        )
        assert_that(span.stop_event, f"Expected non-None span stop event {span}")
        metrics_to_log = _build_metrics([_MetricMapping("app_build_dataiters_finish_time", span.stop_event.timestamp.milliseconds_since_epoch)])
        return metrics_to_log

    def _metrics_for_load_checkpoint_start(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.CHECKPOINT_LOAD,
            "Expected span name to be CHECKPOINT_LOAD",
        )
        metrics_to_log = _build_metrics([_MetricMapping("load_checkpoint_start_time", span.start_event.timestamp.milliseconds_since_epoch)])
        return metrics_to_log

    def _metrics_for_load_checkpoint_stop(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.CHECKPOINT_LOAD,
            "Expected span name to be CHECKPOINT_LOAD",
        )
        assert_that(span.stop_event, f"Expected non-None span stop event {span}")
        metrics_to_log = _build_metrics([_MetricMapping("load_checkpoint_finish_time", span.stop_event.timestamp.milliseconds_since_epoch)])
        metrics_to_log.update(
            _build_metrics([_MetricMapping("load_checkpoint_time", span.attributes[StandardSpanAttributeName.DURATION_MSEC].value, coefficient=0.001)])
        )
        return metrics_to_log

    def _metrics_for_optimizer_init_start(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.OPTIMIZER_INIT,
            "Expected span name to be OPTIMIZER_INIT",
        )
        metrics_to_log = _build_metrics([_MetricMapping("app_build_optimizer_start_time", span.start_event.timestamp.milliseconds_since_epoch)])
        return metrics_to_log

    def _metrics_for_optimizer_init_stop(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.OPTIMIZER_INIT,
            "Expected span name to be OPTIMIZER_INIT",
        )
        assert_that(span.stop_event, f"Expected non-None span stop event {span}")
        metrics_to_log = _build_metrics([_MetricMapping("app_build_optimizer_finish_time", span.stop_event.timestamp.milliseconds_since_epoch)])
        return metrics_to_log

    def _metrics_for_training_loop_start(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.TRAINING_LOOP,
            "Expected span name to be TRAINING_LOOP",
        )
        assert_that(
            isinstance(span.attributes, TrainingLoopAttributes),
            f"Expected span attributes to be of type TrainingLoopAttributes but got {type(span.attributes)}",
        )
        attributes = cast(TrainingLoopAttributes, span.attributes)
        return _build_metrics(
            [
                _MetricMapping("app_train_loop_start_time", span.start_event.timestamp.milliseconds_since_epoch),
                _MetricMapping("train_tokens_target", attributes.train_tokens_target),
                _MetricMapping("train_iterations_target", attributes.train_iterations_target),
                _MetricMapping("train_samples_target", attributes.train_samples_target),
                _MetricMapping("train_iterations_start", attributes.train_iterations_start),
                _MetricMapping("train_iterations_end", attributes.train_iterations_start),
                _MetricMapping("train_samples_start", attributes.train_samples_start),
                _MetricMapping("train_samples_end", attributes.train_samples_start),
                _MetricMapping("train_tflop_start", attributes.completed_floating_point_operations_overall, coefficient=1.0 / (10**12)),
            ]
        )

    def _metrics_for_training_loop_stop(self, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.TRAINING_LOOP,
            "Expected span name to be TRAINING_LOOP",
        )
        assert_that(span.stop_event, f"Expected non-None span stop event {span}")
        return _build_metrics([_MetricMapping("app_train_loop_finish_time", span.stop_event.timestamp.milliseconds_since_epoch)])

    def _metrics_for_training_metrics_update_event(self, event: Event, span: Span) -> dict[str, Any]:
        assert_that(
            event.name == StandardTrainingJobEventName.TRAINING_METRICS_UPDATE,
            "Expected event name to be TRAINING_METRICS_UPDATE",
        )
        assert_that(
            isinstance(event.attributes, TrainingMetricsUpdateAttributes),
            f"Expected event attributes to be of type TrainingMetricsUpdateAttributes but got {type(event.attributes)}",
        )
        attributes = cast(TrainingMetricsUpdateAttributes, event.attributes)
        metrics_to_log: dict[str, Any] = _build_metrics(
            [
                _MetricMapping("train_iterations", attributes.num_iterations),
                # In v1, train_iterations_end is off by one. For example, if we start from iteration 0 and finish one iteration,
                # train_iterations_end will be 1 (even though we finsihed iteration "0" and have not started iteration "1" yet).
                _MetricMapping("train_iterations_end", attributes.current_iteration + 1),
                _MetricMapping("train_iterations_time_total", attributes.total_iteration_time_sec),
                _MetricMapping("train_iterations_time_msecs_min", attributes.min_iteration_time_sec, coefficient=1000),
                _MetricMapping("train_iterations_time_msecs_avg", attributes.avg_iteration_time_sec, coefficient=1000),
                _MetricMapping("train_tokens", attributes.train_tokens),
                _MetricMapping("train_tflop_end", attributes.completed_floating_point_operations_overall, coefficient=1.0 / (10**12)),
                _MetricMapping("first_logged_train_iterations_finish_time", attributes.first_logged_train_iterations_finish_timestamp_sec, coefficient=1000),
                _MetricMapping("last_logged_train_iterations_finish_time", attributes.last_logged_train_iterations_finish_timestamp_sec, coefficient=1000),
                _MetricMapping("train_throughput_per_gpu", attributes.train_throughput_per_gpu),
                _MetricMapping("train_throughput_per_gpu_max", attributes.train_throughput_per_gpu_max),
                _MetricMapping("train_tflop", attributes.total_flops if attributes.total_flops > 0 else None, coefficient=1.0 / (10**12)),
                _MetricMapping("train_samples", attributes.num_train_samples),
                _MetricMapping("train_samples_end", attributes.train_samples_start + attributes.num_train_samples),
            ]
        )
        metrics_to_log.update(self._perf_tag_dict(self._training_telemetry_config))
        return metrics_to_log

    def _metrics_for_validation_metrics_update_event(self, event: Event, span: Span) -> dict[str, Any]:
        assert_that(
            event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE,
            "Expected event name to be VALIDATION_METRICS_UPDATE",
        )
        assert_that(
            isinstance(event.attributes, ValidationMetricsUpdateAttributes),
            f"Expected event attributes to be of type ValidationMetricsUpdateAttributes but got {type(event.attributes)}",
        )
        attributes = cast(ValidationMetricsUpdateAttributes, event.attributes)
        return _build_metrics(
            [
                _MetricMapping("validation_iterations_time_total", attributes.total_iteration_time_sec),
                _MetricMapping("validation_iterations_time_msecs_min", attributes.min_iteration_time_sec, coefficient=1000),
                _MetricMapping("validation_iterations_time_msecs_avg", attributes.avg_iteration_time_sec, coefficient=1000),
            ]
        )

    def _metrics_for_checkpoint_save_start(self, span: Span) -> dict[str, Any]:
        assert_that(
            isinstance(span.attributes, CheckpointSaveSpanAttributes),
            f"Expected span attributes to be of type CheckpointSaveSpanAttributes but got {type(span.attributes)}",
        )
        attributes = cast(CheckpointSaveSpanAttributes, span.attributes)

        return _build_metrics(
            [
                # In v1, train_iterations_save_checkpoint_end is off by one. For example, if we start from iteration 0 and finish one iteration
                # before saving the checkpoint, "train_iterations_save_checkpoint_end" will be 1 (even though we finsihed iteration "0"
                # and have not started iteration "1" yet).
                _MetricMapping("train_iterations_save_checkpoint_end", attributes.current_iteration + 1),
                _MetricMapping("save_checkpoint_count", attributes.save_checkpoint_attempt_count),
            ]
        )

    def _metrics_for_save_checkpoint_success_event(self, event: Event, span: Span) -> dict[str, Any]:
        assert_that(
            event.name == StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS,
            "Expected event name to be SAVE_CHECKPOINT_SUCCESS",
        )
        assert_that(
            isinstance(event.attributes, SaveCheckpointSuccessEventAttributes),
            f"Expected event attributes to be of type SaveCheckpointSuccessAttributes but got {type(event.attributes)}",
        )
        attributes = cast(SaveCheckpointSuccessEventAttributes, event.attributes)
        metrics_to_log = _build_metrics(
            [
                _MetricMapping("first_saved_train_iterations_start_time", attributes.training_start_timestamp_sec, coefficient=1000),
                _MetricMapping("train_iterations_productive_end", attributes.productive_train_iterations),
                _MetricMapping("train_samples_productive_end", attributes.productive_train_samples),
                _MetricMapping("train_tflop_productive_end", attributes.productive_train_tflops),
                _MetricMapping("train_iterations_time_total_productive", attributes.productive_train_iterations_sec),
                _MetricMapping("validation_iterations_time_total_productive", attributes.productive_validation_iterations_sec),
            ]
        )
        if self._training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            # In v1, we only report the first/last save times for sync checkpoints.
            metrics_to_log.update(
                _build_metrics(
                    [
                        _MetricMapping(
                            "first_successful_save_checkpoint_sync_finish_time", attributes.first_successful_save_checkpoint_timestamp_sec, coefficient=1000
                        ),
                        _MetricMapping(
                            "last_successful_save_checkpoint_sync_finish_time", attributes.latest_successful_save_checkpoint_timestamp_sec, coefficient=1000
                        ),
                        _MetricMapping("save_checkpoint_sync_count", attributes.save_checkpoint_success_count),
                    ]
                )
            )
        elif self._training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            metrics_to_log.update(
                _build_metrics(
                    [
                        _MetricMapping("save_checkpoint_async_count", attributes.save_checkpoint_success_count),
                    ]
                )
            )
        return metrics_to_log

    def _metrics_for_sync_checkpoint_metrics_update_event(self, event: Event, span: Span) -> dict[str, Any]:
        assert_that(
            span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC,
            "SYNC_CHECKPOINT_METRICS_UPDATE events are only expected for CHECKPOINT_SAVE_SYNC spans",
        )
        assert_that(
            event.name == StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE,
            "Expected event name to be SYNC_CHECKPOINT_METRICS_UPDATE",
        )
        assert_that(
            isinstance(event.attributes, SyncCheckpointMetricsUpdateAttributes),
            f"Expected event attributes to be of type SyncCheckpointMetricsUpdateAttributes but got {type(event.attributes)}",
        )
        attributes = cast(SyncCheckpointMetricsUpdateAttributes, event.attributes)
        metrics_to_log = _build_metrics(
            [
                _MetricMapping("save_checkpoint_sync_time_total", attributes.save_checkpoint_sync_time_total_sec),
                _MetricMapping("save_checkpoint_sync_time_total_productive", attributes.save_checkpoint_sync_time_total_sec),
                _MetricMapping("save_checkpoint_sync_time_min", attributes.save_checkpoint_sync_time_min_sec),
                _MetricMapping("save_checkpoint_sync_time_max", attributes.save_checkpoint_sync_time_max_sec),
            ]
        )
        return metrics_to_log

    def _get_slurm_metrics(self) -> dict[str, Any]:
        return {
            "jobid": os.environ.get("SLURM_JOBID", None),
            "job_account": os.environ.get("SLURM_JOB_ACCOUNT", None),
            "alloc_hosts": os.environ.get("SLURM_JOB_NUM_NODES", None),
            "cluster": os.environ.get("SLURM_CLUSTER_NAME", None),
            "job_start_time": int(os.environ.get("SLURM_JOB_START_TIME", 0)) * 1000,
            "exec_hosts": os.environ.get("SLURM_JOB_NODELIST", None),
            "user": os.environ.get("SLURM_JOB_USER", None),
            "job_name": os.environ.get("SLURM_JOB_NAME", None),
            "limit_run": (int(os.environ.get("SLURM_JOB_END_TIME", 0)) - int(os.environ.get("SLURM_JOB_START_TIME", 0))),
            "job_partition": os.environ.get("SLURM_JOB_PARTITION", None),
            "job_reservation": os.environ.get("SLURM_JOB_RESERVATION", None),
            "one_logger_version": get_version("nv_one_logger.training_telemetry"),
            "app_first_log_time": round(time.time() * 1000.0),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", None),
            "array_task_count": os.environ.get("SLURM_ARRAY_TASK_COUNT", None),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", None),
            "step_id": os.environ.get("SLURM_STEP_ID")
            or os.environ.get("SLURM_STEPID"),  # For backward compatibility, will be None if neither of these two variables exist
            "step_nodelist": os.environ.get("SLURM_STEP_NODELIST", None),
        }

    def _metrics_for_app_start(self, span: Span) -> dict[str, Any]:
        metrics = _build_metrics([_MetricMapping("app_start_time", span.start_event.timestamp.milliseconds_since_epoch)])
        for key, value in self._get_slurm_metrics().items():
            if value is not None:
                metrics[key] = value
        return metrics

    def _metrics_for_app_stop(self, span: Span) -> dict[str, Any]:
        return _build_metrics([_MetricMapping("app_finish_time", span.stop_event.timestamp.milliseconds_since_epoch)])


class V1CompatibleWandbExporterSync(WandBExporterSync):
    """An exporter that turns v2 telemetry data into v1-compatible metrics format and logs them to wandb synchronously.

    This class ensures compatibility of the metrics with v1. That is, it allows a user to switch to v2 without the need to change the
    downstream consumers of the telemetry data (e.g., data infra that processes the metrics stored in wandb).
    """

    def __init__(
        self,
        training_telemetry_config: TrainingTelemetryConfig,
        wandb_config: WandBConfig,
    ):
        super().__init__(
            config=wandb_config,
            metric_naming_strategy=HierarchicalMetricNamingStrategy(),
        )
        self.adapter = V1CompatibleWandbExporterAdapter(training_telemetry_config)

    @override
    def export_start(self, span: Span) -> None:
        metrics = self.adapter.extract_v1_metrics_for_span_start(span)
        self._log_metrics(metrics)

    @override
    def export_stop(self, span: Span) -> None:
        metrics = self.adapter.extract_v1_metrics_for_span_stop(span)
        self._log_metrics(metrics)

    @override
    def export_event(self, event: Event, span: Span) -> None:
        metrics = self.adapter.extract_v1_metrics_for_event(event, span)
        self._log_metrics(metrics)

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        pass


class V1CompatibleWandbExporterAsync(WandBExporterAsync):
    """An exporter that turns v2 telemetry data into v1-compatible metrics format and logs them to wandb asynchronously.

    This class ensures compatibility of the metrics with v1. That is, it allows a user to switch to v2 without the need to change the
    downstream consumers of the telemetry data (e.g., data infra that processes the metrics stored in wandb).
    """

    def __init__(
        self,
        training_telemetry_config: TrainingTelemetryConfig,
        wandb_config: WandBConfig,
    ):
        super().__init__(
            config=wandb_config,
            metric_naming_strategy=HierarchicalMetricNamingStrategy(),
        )
        self.adapter = V1CompatibleWandbExporterAdapter(training_telemetry_config)

    @override
    def export_start(self, span: Span) -> None:
        metrics = self.adapter.extract_v1_metrics_for_span_start(span)
        self._log_metrics(metrics)

    @override
    def export_stop(self, span: Span) -> None:
        metrics = self.adapter.extract_v1_metrics_for_span_stop(span)
        self._log_metrics(metrics)

    @override
    def export_event(self, event: Event, span: Span) -> None:
        metrics = self.adapter.extract_v1_metrics_for_event(event, span)
        self._log_metrics(metrics)

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        pass


class V1CompatibleExporter:
    """Factory class to create V1CompatibleWandbExporter instances.
    
    This class provides a unified interface for creating v1-compatible wandb exporters
    that can work with either sync or async modes.
    """
    
    def __init__(self, training_telemetry_config: TrainingTelemetryConfig, async_mode: bool = False):
        """Initialize the V1CompatibleExporter.
        
        Args:
            training_telemetry_config: The training telemetry configuration.
            async_mode: If True, creates an async exporter. If False, creates a sync exporter.
        """
        self._training_telemetry_config = training_telemetry_config
        self._async_mode = async_mode
        
        # Create the appropriate exporter config using v1-style configuration
        self._exporter_config = WandBConfig(
            host="https://api.wandb.ai",
            api_key="",
            project=training_telemetry_config.application_name,
            run_name=f"{training_telemetry_config.application_name}-run-{str(uuid.uuid4())}",
            entity="hwinf_dcm",  # NOTE: should always be 'hwinf_dcm' for internal user.
            tags=["e2e_metrics_enabled"],
            save_dir="./wandb",
        )
        
        # Create the appropriate exporter based on async mode
        if async_mode:
            self._exporter = V1CompatibleWandbExporterAsync(
                training_telemetry_config=training_telemetry_config,
                wandb_config=self._exporter_config,
            )
        else:
            self._exporter = V1CompatibleWandbExporterSync(
                training_telemetry_config=training_telemetry_config,
                wandb_config=self._exporter_config,
            )
    
    @property
    def exporter(self):
        """Get the underlying exporter instance."""
        return self._exporter
    
    @property
    def is_async(self):
        """Check if the exporter is in async mode."""
        return self._async_mode
