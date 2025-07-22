# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
import socket
from typing import Generator
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
from nv_one_logger.api.config import ApplicationType
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event
from nv_one_logger.core.internal.singleton import SingletonMeta
from nv_one_logger.core.span import Span, StandardSpanName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter
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
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.wandb.exporter.wandb_exporter import Config as WandBConfig

from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import V1CompatibleWandbExporterAdapter

START_TIME_SEC = 3_500_000
START_PERF_COUNTER = 2000


@pytest.fixture(autouse=True, scope="function")
def reset_v2_provder():
    """Reset the v2 provider singleton to isolate tests."""
    OneLoggerProvider.instance()._config = None  # type: ignore
    OneLoggerProvider.instance()._recorder = None  # type: ignore


@pytest.fixture
def mock_time() -> Generator[mock.Mock, None, None]:
    """Patch time.time and provide the corresponding mock."""
    with mock.patch("time.time") as mock_time:
        mock_time.return_value = START_TIME_SEC
        yield mock_time


@pytest.fixture
def mock_perf_counter() -> Generator[mock.Mock, None, None]:
    """Patch time.perf_counter and provide the corresponding mock."""
    with mock.patch("time.perf_counter") as mock_perf_counter:
        mock_perf_counter.return_value = START_PERF_COUNTER
        yield mock_perf_counter


def advance_time(mock_time: mock.Mock, mock_perf_counter: mock.Mock, seconds: float) -> TracingTimestamp:
    """Advances the mock time by the specified number of seconds and returns a TracingTimestamp corresponding to the new time   ."""
    mock_time.return_value += seconds
    mock_perf_counter.return_value += seconds
    return TracingTimestamp.for_timestamp(
        timestamp_sec=mock_time.return_value,
        perf_counter=mock_perf_counter.return_value,
        validate_timestamp=False,
    )


@pytest.fixture
def mock_exporter() -> Generator[Exporter, None, None]:
    """Fixture that sets up a mock exporter."""
    exporter = MagicMock(spec=Exporter)

    yield exporter

    exporter.reset_mock()


@pytest.fixture(autouse=True)
def configure_provider(
    config: TrainingTelemetryConfig,
    mock_exporter: Exporter,
) -> None:
    """Fixture that configures the TrainingTelemetryProvider."""
    # Reset the state of the singletons
    # Reset the state of the singletons
    with SingletonMeta._lock:
        SingletonMeta._instances.pop(TrainingTelemetryProvider, None)
        SingletonMeta._instances.pop(OneLoggerProvider, None)
    TrainingTelemetryProvider.instance().with_base_telemetry_config(config).with_exporter(mock_exporter).configure_provider()


@pytest.fixture
def adapter(config: TrainingTelemetryConfig) -> V1CompatibleWandbExporterAdapter:
    """Create a V1CompatibleWandbExporterAdapter instance."""
    return V1CompatibleWandbExporterAdapter(config)


class TestV1CompatibleWandbExporterAdapter:
    """Test cases for the V1CompatibleWandbExporterAdapter class."""

    def test_perf_tag_dict_(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
    ) -> None:
        """Test _perf_tag_dict method with string perf_tag."""
        assert adapter._training_telemetry_config.training_loop_config is not None
        adapter._training_telemetry_config.training_loop_config.perf_tag_or_fn = "test_perf"
        result = adapter._perf_tag_dict(adapter._training_telemetry_config)

        assert result["app_tag"] == ["test_perf"]
        assert len(result["app_tag_id"]) == 1
        assert result["app_tag_count"] == 1

        adapter._training_telemetry_config.training_loop_config.perf_tag_or_fn = ["perf1", "perf2"]
        result = adapter._perf_tag_dict(adapter._training_telemetry_config)

        assert result["app_tag"] == ["perf1", "perf2"]
        assert len(result["app_tag_id"]) == 2
        assert result["app_tag_count"] == 2

    def test_application_span(
        self,
        config: TrainingTelemetryConfig,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test the extracted metrics for APPLICATION span and its ONE_LOGGER_INITIALIZATION event."""
        app_span = Span.create(
            name=StandardSpanName.APPLICATION,
            span_attributes=Attributes({}),
        )

        span_start_metrics = adapter.extract_v1_metrics_for_span_start(app_span)

        assert span_start_metrics == {
            "app_start_time": START_TIME_SEC * 1000,
            "app_first_log_time": START_TIME_SEC * 1000,
            "job_start_time": 0,
            "limit_run": 0,
            "one_logger_version": "2.0.1",
        }

        one_logger_init_attributes = OneLoggerInitializationAttributes.create(
            one_logger_training_telemetry_version="2.0.1",
            enable_for_current_rank=True,
            session_tag="fake_session_tag",
            app_type=ApplicationType.TRAINING,
            is_baseline_run=False,
            is_train_iterations_enabled=True,
            is_validation_iterations_enabled=True,
            is_test_iterations_enabled=True,
            is_save_checkpoint_enabled=True,
            is_log_throughput_enabled=True,
            node_name=socket.gethostname(),
            rank=0,
            checkpoint_strategy=CheckPointStrategy.SYNC,
            summary_data_schema_version="1.0",
            custom_metadata={"app_tag_run_version": "2.3.4"},
        )
        advance_time(mock_time, mock_perf_counter, 10)
        event = Event.create(
            name=StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION,
            attributes=one_logger_init_attributes,
        )
        app_span.add_event(event)

        initialization_event_metrics = adapter.extract_v1_metrics_for_event(event=event, span=app_span)
        assert initialization_event_metrics == {
            "app_tag_run_name": one_logger_init_attributes.session_tag,
            "summary_data_schema_version": "1.0",
            "app_run_type": ApplicationType.TRAINING,
            "app_metrics_feature_tags": "full",
            "is_baseline_run": False,
            "is_train_iterations_enabled": one_logger_init_attributes.is_train_iterations_enabled,
            "is_validation_iterations_enabled": one_logger_init_attributes.is_validation_iterations_enabled,
            "is_test_iterations_enabled": one_logger_init_attributes.is_test_iterations_enabled,
            "is_save_checkpoint_enabled": one_logger_init_attributes.is_save_checkpoint_enabled,
            "is_log_throughput_enabled": one_logger_init_attributes.is_log_throughput_enabled,
            "app_tag_run_version": "2.3.4",
            "save_checkpoint_strategy": "sync",
        }

        advance_time(mock_time, mock_perf_counter, 50)
        app_span.stop(
            stop_event_attributes=Attributes({}),
        )
        span_stop_metrics = adapter.extract_v1_metrics_for_span_stop(app_span)
        assert span_stop_metrics == {
            "app_finish_time": (START_TIME_SEC + 60) * 1000,
        }

    def test_model_init_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for MODEL_INIT span."""
        model_init_span = Span.create(
            name=StandardTrainingJobSpanName.MODEL_INIT,
        )

        span_start_metrics = adapter.extract_v1_metrics_for_span_start(model_init_span)

        assert span_start_metrics == {"app_model_init_start_time": START_TIME_SEC * 1000}

        advance_time(mock_time, mock_perf_counter, 10)
        model_init_span.stop()

        span_stop_metrics = adapter.extract_v1_metrics_for_span_stop(model_init_span)
        assert span_stop_metrics == {
            "app_model_init_finish_time": (START_TIME_SEC + 10) * 1000,
        }

    def test_data_loader_init_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for DATA_LOADER_INIT span."""
        dataloader_init_span = Span.create(name=StandardTrainingJobSpanName.DATA_LOADER_INIT)

        span_start_metrics = adapter.extract_v1_metrics_for_span_start(dataloader_init_span)

        assert span_start_metrics == {"app_build_dataiters_start_time": START_TIME_SEC * 1000}

        advance_time(mock_time, mock_perf_counter, 10)
        dataloader_init_span.stop()

        span_stop_metrics = adapter.extract_v1_metrics_for_span_stop(dataloader_init_span)
        assert span_stop_metrics == {
            "app_build_dataiters_finish_time": (START_TIME_SEC + 10) * 1000,
        }

    def test_checkpoint_load_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for CHECKPOINT_LOAD span."""
        checkpoint_load_span = Span.create(
            name=StandardTrainingJobSpanName.CHECKPOINT_LOAD,
        )

        span_start_metrics = adapter.extract_v1_metrics_for_span_start(checkpoint_load_span)

        assert span_start_metrics == {"load_checkpoint_start_time": START_TIME_SEC * 1000}

        advance_time(mock_time, mock_perf_counter, 10)
        checkpoint_load_span.stop()
        span_stop_metrics = adapter.extract_v1_metrics_for_span_stop(checkpoint_load_span)
        assert span_stop_metrics == {
            "load_checkpoint_finish_time": (START_TIME_SEC + 10) * 1000,
            "load_checkpoint_time": 10,  # 10 seconds
        }

    def test_optimizer_init_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for OPTIMIZER_INIT span."""
        optimizer_init_span = Span.create(
            name=StandardTrainingJobSpanName.OPTIMIZER_INIT,
        )

        span_start_metrics = adapter.extract_v1_metrics_for_span_start(optimizer_init_span)

        assert span_start_metrics == {"app_build_optimizer_start_time": START_TIME_SEC * 1000}

        advance_time(mock_time, mock_perf_counter, 10)
        optimizer_init_span.stop()
        span_stop_metrics = adapter.extract_v1_metrics_for_span_stop(optimizer_init_span)
        assert span_stop_metrics == {
            "app_build_optimizer_finish_time": (START_TIME_SEC + 10) * 1000,
        }

    def test_training_loop_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for TRAINING_LOOP span and its TRAINING_METRICS_UPDATE event."""
        training_loop_attributes = TrainingLoopAttributes.create(
            perf_tag="fake_perf_tag",
            log_every_n_train_iterations=10,
            world_size=10,
            global_batch_size=32,
            train_iterations_start=10,
            train_samples_start=320,
            train_iterations_target=1000,
            train_samples_target=32000,
            train_tokens_target=1024 * 32000,
            micro_batch_size=1,
            seq_length=512,
        )
        training_loop_span = Span.create(
            name=StandardTrainingJobSpanName.TRAINING_LOOP,
            span_attributes=training_loop_attributes,
        )

        metrics = adapter.extract_v1_metrics_for_span_start(training_loop_span)

        assert metrics == {
            "app_tag": ["test_perf"],
            "app_tag_id": ["2eee30ac275fc16962ae8d2472a7b68d"],
            "app_tag_count": 1,
            "world_size": 10,
            "global_batch_size": 32,
            "train_iterations_start": 10,
            "train_iterations_end": 10,
            "train_samples_start": 320,
            "train_samples_end": 320,
            "train_iterations_target": 1000,
            "train_samples_target": 32000,
            "app_train_loop_start_time": START_TIME_SEC * 1000,
            "train_tokens_target": 1024 * 32000,
            "micro_batch_size": 1,
            "model_seq_length": 512,
        }

        advance_time(mock_time, mock_perf_counter, 50)
        training_iteration_span = Span.create(name=StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION)
        metrics = adapter.extract_v1_metrics_for_span_start(training_iteration_span)
        # No metrics created for the TRAINING_SINGLE_ITERATION span start.
        assert metrics == {}

        advance_time(mock_time, mock_perf_counter, 200)
        training_iteration_span.stop()
        metrics = adapter.extract_v1_metrics_for_span_stop(training_iteration_span)
        # No metrics created for the TRAINING_SINGLE_ITERATION span start.
        assert metrics == {}

        training_metrics_update_attributes = TrainingMetricsUpdateAttributes.create(
            train_iterations_start=10,
            current_iteration=16,
            num_iterations=7,
            train_samples_start=320,
            num_train_samples=7 * 32,
            interval=8,
            avg_iteration_time_sec=50.0,
            min_iteration_time_sec=20.0,
            max_iteration_time_sec=90.0,
            total_iteration_time_sec=50.0 * 7,
            train_tokens=1024 * 7 * 32,
            # 10 iterations in the previous run (iterations 0 upto and incl 9 and 7 iterations in the current job)
            completed_floating_point_operations_overall=(10 + 7) * 32 * 100,
            total_flops=32 * 100 * 7,
            train_throughput_per_gpu=32 * 100.0 / (50.0 * 10**12 * 10),
            train_throughput_per_gpu_max=32 * 100.0 / (50.0 * 90**12 * 10),
            train_throughput_per_gpu_min=32 * 100.0 / (50.0 * 20**12 * 10),
            first_logged_train_iterations_finish_timestamp_sec=START_TIME_SEC + 400,
            last_logged_train_iterations_finish_timestamp_sec=START_TIME_SEC + 6000,
        )

        advance_time(mock_time, mock_perf_counter, 400)
        event = Event.create(
            name=StandardTrainingJobEventName.TRAINING_METRICS_UPDATE,
            attributes=training_metrics_update_attributes,
        )
        training_loop_span.add_event(event)
        metrics = adapter.extract_v1_metrics_for_event(event=event, span=training_loop_span)
        assert metrics == {
            "app_tag": ["test_perf"],
            "app_tag_id": ["2eee30ac275fc16962ae8d2472a7b68d"],
            "app_tag_count": 1,
            "train_iterations": 7,
            # This value is off by one in v1.
            "train_iterations_end": 17,
            "train_iterations_time_total": 350.0,
            "train_iterations_time_msecs_min": 20_000.0,
            "train_iterations_time_msecs_avg": 50_000.0,
            "train_tokens": 1024 * 7 * 32,
            "train_throughput_per_gpu": 32 * 100.0 / (50.0 * 10**12 * 10),
            "train_throughput_per_gpu_max": 32 * 100.0 / (50.0 * 90**12 * 10),
            "train_tflop": float(32 * 100 * 7) / (10**12),
            "train_tflop_end": float((10 + 7) * 32 * 100) / (10**12),
            "train_samples": 32 * 7,
            "train_samples_end": 32 * 17,
            "first_logged_train_iterations_finish_time": (START_TIME_SEC + 400) * 1000,
            "last_logged_train_iterations_finish_time": (START_TIME_SEC + 6000) * 1000,
        }

        advance_time(mock_time, mock_perf_counter, 500)
        training_loop_span.stop()
        metrics = adapter.extract_v1_metrics_for_span_stop(training_loop_span)
        assert metrics == {
            "app_train_loop_finish_time": (START_TIME_SEC + 50 + 200 + 400 + 500) * 1000,
        }

    def test_validation_loop(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for VALIDATION_LOOP span and its VALIDATION_METRICS_UPDATE event."""
        validation_loop_span = Span.create(
            name=StandardTrainingJobSpanName.VALIDATION_LOOP,
        )

        metrics = adapter.extract_v1_metrics_for_span_start(validation_loop_span)

        assert metrics == {}

        advance_time(mock_time, mock_perf_counter, 50)
        validation_iteration_span = Span.create(name=StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION)
        metrics = adapter.extract_v1_metrics_for_span_start(validation_iteration_span)
        # No metrics created for the VALIDATION_SINGLE_ITERATION span start.
        assert metrics == {}

        advance_time(mock_time, mock_perf_counter, 200)
        validation_iteration_span.stop()
        metrics = adapter.extract_v1_metrics_for_span_stop(validation_iteration_span)
        # No metrics created for the VALIDATION_SINGLE_ITERATION span start.
        assert metrics == {}

        validation_metrics_update_attributes = ValidationMetricsUpdateAttributes.create(
            current_iteration=16,
            interval=8,
            avg_iteration_time_sec=50.0,
            min_iteration_time_sec=20.0,
            max_iteration_time_sec=90.0,
            total_iteration_time_sec=50.0 * 8,
        )

        advance_time(mock_time, mock_perf_counter, 400)
        event = Event.create(
            name=StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE,
            attributes=validation_metrics_update_attributes,
        )
        validation_loop_span.add_event(event)
        metrics = adapter.extract_v1_metrics_for_event(event=event, span=validation_loop_span)
        assert metrics == {
            "validation_iterations_time_total": 50.0 * 8,
            "validation_iterations_time_msecs_min": 20_000.0,
            "validation_iterations_time_msecs_avg": 50_000.0,
        }

        advance_time(mock_time, mock_perf_counter, 500)
        validation_loop_span.stop()
        metrics = adapter.extract_v1_metrics_for_span_stop(validation_loop_span)
        assert metrics == {}

    def test_sync_checkpoint_save_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for CHECKPOINT_SAVE_SYNC span and its SAVE_CHECKPOINT_SUCCESS and SYNC_CHECKPOINT_METRICS_UPDATE events."""
        span_attributes = CheckpointSaveSpanAttributes.create(
            checkpoint_strategy=CheckPointStrategy.SYNC,
            current_iteration=20,
            save_checkpoint_attempt_count=3,
        )
        checkpoint_save_span = Span.create(
            name=StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC,
            span_attributes=span_attributes,
        )
        metrics = adapter.extract_v1_metrics_for_span_start(checkpoint_save_span)
        assert metrics == {
            # This value is off by one in v1.
            "train_iterations_save_checkpoint_end": 21,
            "save_checkpoint_count": 3,
        }
        advance_time(mock_time, mock_perf_counter, 70)
        success_event = Event.create(
            name=StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS,
            attributes=SaveCheckpointSuccessEventAttributes.create(
                checkpoint_strategy=CheckPointStrategy.SYNC,
                current_iteration=20,
                save_checkpoint_success_count=3,
                first_successful_save_checkpoint_timestamp_sec=222_000.0,
                latest_successful_save_checkpoint_timestamp_sec=230_000.0,
                training_start_timestamp_sec=5_000.0,
                productive_train_iterations=20,
                productive_train_samples=200,
                productive_train_iterations_sec=40000.0,
                productive_validation_iterations_sec=10000.0,
            ),
        )

        checkpoint_save_span.add_event(success_event)
        metrics = adapter.extract_v1_metrics_for_event(event=success_event, span=checkpoint_save_span)
        assert metrics == {
            "save_checkpoint_sync_count": 3,
            "first_successful_save_checkpoint_sync_finish_time": 222_000_000,
            "last_successful_save_checkpoint_sync_finish_time": 230_000_000,
            "first_saved_train_iterations_start_time": 5_000_000,
            "train_iterations_productive_end": 20,
            "train_samples_productive_end": 200,
            "train_iterations_time_total_productive": 40000.0,
            "validation_iterations_time_total_productive": 10000.0,
        }
        metrics_update_event = Event.create(
            name=StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE,
            attributes=SyncCheckpointMetricsUpdateAttributes.create(
                save_checkpoint_sync_time_total_sec=550.0,
                save_checkpoint_sync_time_min_sec=20.0,
                save_checkpoint_sync_time_max_sec=120.0,
            ),
        )
        checkpoint_save_span.add_event(metrics_update_event)
        metrics = adapter.extract_v1_metrics_for_event(event=metrics_update_event, span=checkpoint_save_span)
        assert metrics == {
            "save_checkpoint_sync_time_total": 550,
            "save_checkpoint_sync_time_total_productive": 550.0,
            "save_checkpoint_sync_time_min": 20,
            "save_checkpoint_sync_time_max": 120,
        }

        advance_time(mock_time, mock_perf_counter, 10)
        checkpoint_save_span.stop()
        metrics = adapter.extract_v1_metrics_for_span_stop(checkpoint_save_span)
        assert metrics == {}

    def test_async_checkpoint_save_span(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
        mock_time: Mock,
        mock_perf_counter: Mock,
    ) -> None:
        """Test the extracted metrics for CHECKPOINT_SAVE_ASYNC span and its SAVE_CHECKPOINT_SUCCESS events."""
        span_attributes = CheckpointSaveSpanAttributes.create(
            checkpoint_strategy=CheckPointStrategy.ASYNC,
            current_iteration=20,
            save_checkpoint_attempt_count=3,
        )
        checkpoint_save_span = Span.create(
            name=StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC,
            span_attributes=span_attributes,
        )
        metrics = adapter.extract_v1_metrics_for_span_start(checkpoint_save_span)
        assert metrics == {
            # This value is off by one in v1.
            "train_iterations_save_checkpoint_end": 21,
            "save_checkpoint_count": 3,
        }
        advance_time(mock_time, mock_perf_counter, 70)
        success_event = Event.create(
            name=StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS,
            attributes=SaveCheckpointSuccessEventAttributes.create(
                checkpoint_strategy=CheckPointStrategy.ASYNC,
                current_iteration=20,
                save_checkpoint_success_count=3,
                first_successful_save_checkpoint_timestamp_sec=222_000.0,
                latest_successful_save_checkpoint_timestamp_sec=230_000.0,
                training_start_timestamp_sec=5_000.0,
                productive_train_iterations=20,
                productive_train_samples=200,
                productive_train_iterations_sec=40000.0,
                productive_validation_iterations_sec=10000.0,
            ),
        )

        checkpoint_save_span.add_event(success_event)
        metrics = adapter.extract_v1_metrics_for_event(event=success_event, span=checkpoint_save_span)
        assert metrics == {
            "first_successful_save_checkpoint_sync_finish_time": 222_000_000,
            "last_successful_save_checkpoint_sync_finish_time": 230_000_000,
            "first_saved_train_iterations_start_time": 5_000_000,
            "save_checkpoint_sync_count": 3,
            "train_iterations_productive_end": 20,
            "train_samples_productive_end": 200,
            "train_iterations_time_total_productive": 40000.0,
            "validation_iterations_time_total_productive": 10000.0,
        }
        advance_time(mock_time, mock_perf_counter, 10)
        checkpoint_save_span.stop()
        metrics = adapter.extract_v1_metrics_for_span_stop(checkpoint_save_span)
        assert metrics == {}

    def test_ignore_other_spans(
        self,
        adapter: V1CompatibleWandbExporterAdapter,
    ) -> None:
        """Test that the v1 compatible wandb exporter produces metrics only for the spans that are known to have corresponding v1 metrics."""
        spans_with_v1_metrics = [
            StandardSpanName.APPLICATION,
            StandardTrainingJobSpanName.MODEL_INIT,
            StandardTrainingJobSpanName.DATA_LOADER_INIT,
            StandardTrainingJobSpanName.CHECKPOINT_LOAD,
            StandardTrainingJobSpanName.OPTIMIZER_INIT,
            StandardTrainingJobSpanName.TRAINING_LOOP,
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC,
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC,
        ]
        for span_name in StandardTrainingJobSpanName:
            if span_name not in spans_with_v1_metrics:
                span = Span.create(
                    name=span_name,
                    span_attributes=Attributes({"dummy_span_attr": "dummy_span_attr_value"}),
                    start_event_attributes=Attributes({"dummy_start_event_attr": "dummy_start_event_attr_value"}),
                )
                assert adapter.extract_v1_metrics_for_span_start(span) == {}

                span.stop(
                    stop_event_attributes=Attributes({"dummy_stop_event_attr": "dummy_stop_event_attr_value"}),
                )
                assert adapter.extract_v1_metrics_for_span_stop(span) == {}


@pytest.fixture
def wandb_config() -> WandBConfig:
    """Create Dummy wandb config."""
    return WandBConfig(entity="test_entity", project="test_project")
