# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the callbacks module."""

import socket
from typing import Dict, Union
from unittest.mock import MagicMock, Mock

import pytest
from nv_one_logger.api.config import ApplicationType
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.attributes import Attributes, AttributeValue
from nv_one_logger.core.event import StandardEventAttributeName
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.span import StandardSpanAttributeName, StandardSpanName
from nv_one_logger.exporter.exporter import Exporter

from nv_one_logger.training_telemetry.api.attributes import (
    CheckpointSaveSpanAttributes,
    SaveCheckpointSuccessEventAttributes,
    SyncCheckpointMetricsUpdateAttributes,
    TestingMetricsUpdateAttributes,
    TrainingLoopAttributes,
    TrainingMetricsUpdateAttributes,
    ValidationMetricsUpdateAttributes,
)
from nv_one_logger.training_telemetry.api.callbacks import (
    on_app_end,
    on_app_start,
    on_dataloader_init_end,
    on_dataloader_init_start,
    on_load_checkpoint_end,
    on_load_checkpoint_start,
    on_model_init_end,
    on_model_init_start,
    on_optimizer_init_end,
    on_optimizer_init_start,
    on_save_checkpoint_end,
    on_save_checkpoint_start,
    on_save_checkpoint_success,
    on_testing_end,
    on_testing_start,
    on_train_end,
    on_train_start,
    on_training_single_iteration_end,
    on_training_single_iteration_start,
    on_validation_end,
    on_validation_single_iteration_end,
    on_validation_single_iteration_start,
    on_validation_start,
)
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from .conftest import configure_provider_for_test, reconfigure_provider
from .utils import (
    advance_time,
    all_events_from_export_event,
    assert_exporter_method_call_sequence,
    assert_has_stop_event,
    assert_only_start_event,
    assert_only_start_stop_event,
    event_from_export_event,
    get_non_trivial_events,
    reset_singletong_providers_for_test,
    span_from_export_start,
    span_from_export_stop,
)


@pytest.fixture(autouse=True)
def configure_provider(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
    """Fixture that configures the TrainingTelemetryProvider."""
    configure_provider_for_test(config, mock_exporter)


STARTING_PERF_COUNTER = 5000.0
STARTING_TIME = 120000.0


@pytest.fixture(autouse=True)
def initialize_time(mock_time: Mock, mock_perf_counter: Mock) -> None:
    """Initialize the time and perf counter mocks."""
    mock_time.return_value = STARTING_TIME
    mock_perf_counter.return_value = STARTING_PERF_COUNTER


@pytest.mark.parametrize(
    "start_time_msec,finish_time_msec",
    [
        (None, None),
        ((STARTING_TIME - 40) * 1000, None),
        ((STARTING_TIME - 40) * 1000, (STARTING_TIME - 2) * 1000),
    ],
    ids=["no_times", "start_time_only", "both_times"],
)
def test_app_lifecycle_callbacks(
    mock_exporter: MagicMock,
    mock_perf_counter: Mock,
    mock_time: Mock,
    start_time_msec: Union[float, None],
    finish_time_msec: Union[float, None],
) -> None:
    """Test that application lifecycle callbacks create and stop the appropriate spans and events.

    Args:
        mock_exporter: Mocked exporter instance
        mock_perf_counter: Mocked perf counter
        mock_time: Mocked time
        start_time_msec: Optional start time in milliseconds. If None, current time will be used.
        end_time_msec: Optional end time in milliseconds. If None, current time will be used.
    """
    on_app_start(start_time_msec)

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardSpanName.APPLICATION
    assert span.attributes == Attributes({})

    # Verify initialization event
    assert mock_exporter.export_event.call_count == 1
    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION
    expected_attributes: Dict[str, AttributeValue] = {
        StandardEventAttributeName.TIMESTAMP_MSEC: start_time_msec if start_time_msec is not None else STARTING_TIME * 1000,
        "one_logger_training_telemetry_version": "2.0.1",
        "enable_for_current_rank": True,
        "session_tag": "test_session",
        "app_type": ApplicationType.TRAINING,
        "is_baseline_run": False,
        "is_train_iterations_enabled": True,
        "is_validation_iterations_enabled": True,
        "is_test_iterations_enabled": True,
        "is_save_checkpoint_enabled": True,
        "is_log_throughput_enabled": True,
        "summary_data_schema_version": "1.0.0",
        "node_name": socket.gethostname(),
        "rank": 0,
        "checkpoint_strategy": CheckPointStrategy.SYNC,
    }
    assert event.attributes == Attributes(expected_attributes)
    # Make sure the event exported is the same as the event stored in the span.
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 1
    assert span_events[0] == event

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_app_end(finish_time_msec)

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardSpanName.APPLICATION

    expected_start = start_time_msec if start_time_msec is not None else STARTING_TIME * 1000
    expected_finish = finish_time_msec if finish_time_msec is not None else (STARTING_TIME + 10) * 1000
    expected_duration = expected_finish - expected_start

    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: expected_duration})
    assert_has_stop_event(span)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
            Exporter.close,
        ],
    )


@pytest.mark.parametrize(
    "start_time_msec,finish_time_msec",
    [
        (None, None),
        ((STARTING_TIME - 40) * 1000, None),
        ((STARTING_TIME - 40) * 1000, (STARTING_TIME - 2) * 1000),
    ],
    ids=["no_times", "start_time_only", "both_times"],
)
def test_model_init_callbacks(
    mock_exporter: MagicMock,
    mock_perf_counter: Mock,
    mock_time: Mock,
    start_time_msec: Union[float, None],
    finish_time_msec: Union[float, None],
) -> None:
    """Test that model initialization callbacks create and stop the appropriate spans.

    Args:
        mock_exporter: Mocked exporter instance
        mock_perf_counter: Mocked perf counter
        mock_time: Mocked time
        start_time_msec: Optional start time in milliseconds. If None, current time will be used.
        finish_time_msec: Optional end time in milliseconds. If None, current time will be used.
    """
    on_model_init_start(start_time_msec)

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.MODEL_INIT
    assert span.attributes == Attributes({})
    assert_only_start_event(span)

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_model_init_end(finish_time_msec)

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.MODEL_INIT

    expected_start = start_time_msec if start_time_msec is not None else STARTING_TIME * 1000
    expected_finish = finish_time_msec if finish_time_msec is not None else (STARTING_TIME + 10) * 1000
    expected_duration = expected_finish - expected_start

    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: expected_duration})

    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


@pytest.mark.parametrize(
    "start_time_msec,finish_time_msec",
    [
        (None, None),
        ((STARTING_TIME - 40) * 1000, None),
        ((STARTING_TIME - 40) * 1000, (STARTING_TIME - 2) * 1000),
    ],
    ids=["no_times", "start_time_only", "both_times"],
)
def test_dataloader_init_callbacks(
    mock_exporter: MagicMock,
    mock_perf_counter: Mock,
    mock_time: Mock,
    start_time_msec: Union[float, None],
    finish_time_msec: Union[float, None],
) -> None:
    """Test that dataloader initialization callbacks create and stop the appropriate spans.

    Args:
        mock_exporter: Mocked exporter instance
        mock_perf_counter: Mocked perf counter
        mock_time: Mocked time
        start_time_msec: Optional start time in milliseconds. If None, current time will be used.
        finish_time_msec: Optional end time in milliseconds. If None, current time will be used.
    """
    on_dataloader_init_start(start_time_msec)

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.DATA_LOADER_INIT
    assert span.attributes == Attributes({})

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_dataloader_init_end(finish_time_msec)

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.DATA_LOADER_INIT

    expected_start = start_time_msec if start_time_msec is not None else STARTING_TIME * 1000
    expected_finish = finish_time_msec if finish_time_msec is not None else (STARTING_TIME + 10) * 1000
    expected_duration = expected_finish - expected_start

    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: expected_duration})

    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_training_single_iteration_callbacks(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that training single iteration callbacks create and stop the appropriate spans."""
    on_training_single_iteration_start()
    advance_time(mock_time, mock_perf_counter, 10.0)
    on_training_single_iteration_end()

    # We don't export anything for training batch start/end.
    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
        ],
    )


def test_validation_callbacks_in_isolation(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that validation callbacks create and stop the appropriate spans."""
    on_validation_start()

    # Verify validation loop start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert span.attributes == Attributes({})

    advance_time(mock_time, mock_perf_counter, 10.0)
    for _ in range(30):
        on_validation_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 5.0)
        on_validation_single_iteration_end()

    # Verify validation batch span doesn't get exported.
    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 0

    advance_time(mock_time, mock_perf_counter, 400.0)
    on_validation_end()
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: (10 + 30 * 5 + 400) * 1000})

    # Verify success event
    assert mock_exporter.export_event.call_count == 1
    event = event_from_export_event(mock_exporter, span)
    # Make sure the event exported is the same as the event stored in the span.
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 1
    assert span_events[0] == event
    assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE
    expected_ev_attributes = ValidationMetricsUpdateAttributes.create(
        current_iteration=0, interval=0, avg_iteration_time_sec=5.0, min_iteration_time_sec=5.0, max_iteration_time_sec=5.0, total_iteration_time_sec=150.0
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10 + 30 * 5 + 400) * 1000)
    assert event.attributes == expected_ev_attributes

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_validation_callbacks_within_training_loop(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, config: TrainingTelemetryConfig) -> None:
    """Tests that the validation attributes are updated correctly within the training loop."""
    config.is_log_throughput_enabled_or_fn = False
    reconfigure_provider(config, mock_exporter)

    on_train_start(train_iterations_start=0)
    for _ in range(30):
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 50.0)
        on_training_single_iteration_end()

    span = on_validation_start()
    for _ in range(10):
        on_validation_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 5.0)
        on_validation_single_iteration_end()
    on_validation_end()
    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE
    expected_timestamp_msec = (STARTING_TIME + 30 * 50 + 10 * 5) * 1000
    expected_ev_attributes = ValidationMetricsUpdateAttributes.create(
        current_iteration=29, interval=30, avg_iteration_time_sec=5.0, min_iteration_time_sec=5.0, max_iteration_time_sec=5.0, total_iteration_time_sec=50.0
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, expected_timestamp_msec)
    assert event.attributes == expected_ev_attributes

    for _ in range(40):
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 50.0)
        on_training_single_iteration_end()

    span = on_validation_start()
    for _ in range(10):
        on_validation_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 10.0)
        on_validation_single_iteration_end()
    on_validation_end()

    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE
    expected_ev_attributes = ValidationMetricsUpdateAttributes.create(
        current_iteration=69,
        interval=40,
        avg_iteration_time_sec=7.5,
        min_iteration_time_sec=5.0,
        max_iteration_time_sec=10.0,
        total_iteration_time_sec=50.0 + 100.0,
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, expected_timestamp_msec + (40 * 50 + 10 * 10) * 1000)
    assert event.attributes == expected_ev_attributes


def test_save_sync_checkpoint_callbacks(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, config: TrainingTelemetryConfig) -> None:
    """Test that save sync checkpoint callbacks create and stop the appropriate spans and events."""
    config.save_checkpoint_strategy = CheckPointStrategy.SYNC
    reconfigure_provider(config, mock_exporter)

    global_step = 100
    train_span = on_train_start(
        train_iterations_start=0,
        train_samples_start=0,
        train_iterations_target_or_fn=_dummy_train_iterations_target_fn,
        train_samples_target_or_fn=_dummy_train_samples_target_fn,
    )

    on_save_checkpoint_start(global_step)

    # Verify span start
    assert mock_exporter.export_start.call_count == 2
    span = span_from_export_start(mock_exporter, train_span)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
    assert span.attributes == CheckpointSaveSpanAttributes.create(CheckPointStrategy.SYNC, global_step, 1)

    advance_time(mock_time, mock_perf_counter, 10.0)

    on_save_checkpoint_success(global_step)

    on_save_checkpoint_end()

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 10000})

    # Verify the events
    assert mock_exporter.export_event.call_count == 2
    events = all_events_from_export_event(mock_exporter, span)
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 2

    ckpt_success_event = events[0]
    assert ckpt_success_event.name == StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS
    # Make sure the events exported is the same as the event stored in the span.
    assert span_events[0] == ckpt_success_event
    expected_ev1_attributes = SaveCheckpointSuccessEventAttributes.create(
        checkpoint_strategy=CheckPointStrategy.SYNC,
        current_iteration=global_step,
        first_successful_save_checkpoint_timestamp_sec=STARTING_TIME + 10,
        latest_successful_save_checkpoint_timestamp_sec=STARTING_TIME + 10,
        save_checkpoint_success_count=1,
        productive_train_iterations=0,
        productive_train_samples=0,
        productive_train_iterations_sec=0,
        productive_validation_iterations_sec=0,
        productive_train_tflops=0,
        training_start_timestamp_sec=STARTING_TIME,
    ).add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10) * 1000)

    assert ckpt_success_event.attributes == expected_ev1_attributes

    ckpt_sync_metrics_update_event = events[1]
    assert ckpt_sync_metrics_update_event.name == StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE
    # Make sure the events exported is the same as the event stored in the span.
    assert span_events[1] == ckpt_sync_metrics_update_event
    expected_ev2_attributes = SyncCheckpointMetricsUpdateAttributes.create(
        save_checkpoint_sync_time_total_sec=10.0,
        save_checkpoint_sync_time_min_sec=10.0,
        save_checkpoint_sync_time_max_sec=10.0,
    ).add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10) * 1000)
    assert ckpt_sync_metrics_update_event.attributes == expected_ev2_attributes

    advance_time(mock_time, mock_perf_counter, 500.0)
    span = span_from_export_stop(mock_exporter)
    on_train_end()
    assert mock_exporter.export_stop.call_count == 2
    span = span_from_export_stop(mock_exporter)
    assert span == train_span

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_event,
            Exporter.export_stop,
            Exporter.export_stop,
        ],
    )


def test_save_async_checkpoint_callbacks(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, config: TrainingTelemetryConfig) -> None:
    """Test that save sync checkpoint callbacks create and stop the appropriate spans and events."""
    config.save_checkpoint_strategy = CheckPointStrategy.ASYNC
    reconfigure_provider(config, mock_exporter)

    global_step = 100

    app_span = on_app_start()
    train_span = on_train_start(
        train_iterations_start=0,
        train_samples_start=0,
        train_iterations_target_or_fn=_dummy_train_iterations_target_fn,
        train_samples_target_or_fn=_dummy_train_samples_target_fn,
    )

    on_save_checkpoint_start(global_step)

    # Verify span start
    assert mock_exporter.export_start.call_count == 3
    span = span_from_export_start(mock_exporter, train_span)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC
    assert span.attributes == CheckpointSaveSpanAttributes.create(CheckPointStrategy.ASYNC, global_step, 1)

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_save_checkpoint_end()

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 10000})

    # For async checkpoints, the success event is fired after the checkpoint save span ends and is
    # a child of the application span.
    advance_time(mock_time, mock_perf_counter, 20.0)
    on_save_checkpoint_success(global_step)
    events = get_non_trivial_events(app_span)
    assert len(events) == 2
    checkpoint_success_event = events[1]
    assert checkpoint_success_event.name == StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS
    expected_ev_attributes = SaveCheckpointSuccessEventAttributes.create(
        checkpoint_strategy=CheckPointStrategy.ASYNC,
        current_iteration=global_step,
        first_successful_save_checkpoint_timestamp_sec=STARTING_TIME + 30,
        latest_successful_save_checkpoint_timestamp_sec=STARTING_TIME + 30,
        save_checkpoint_success_count=1,
        productive_train_iterations=0,
        productive_train_samples=0,
        productive_train_iterations_sec=0,
        productive_validation_iterations_sec=0,
        productive_train_tflops=0,
        training_start_timestamp_sec=STARTING_TIME,
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 30) * 1000)
    assert checkpoint_success_event.attributes == expected_ev_attributes

    advance_time(mock_time, mock_perf_counter, 500.0)
    span = span_from_export_stop(mock_exporter)
    on_train_end()
    on_app_end()
    assert mock_exporter.export_stop.call_count == 3
    span = span_from_export_stop(mock_exporter)
    assert span == app_span
    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,  # INITIALIZATION EVENT
            Exporter.export_start,
            Exporter.export_start,
            Exporter.export_stop,
            Exporter.export_event,  # SAVE_CHECKPOINT_SUCCESS EVENT
            Exporter.export_stop,
            Exporter.export_stop,
            Exporter.close,
        ],
    )


def _dummy_train_iterations_target_fn() -> int:
    return 1000


def _dummy_train_samples_target_fn() -> int:
    return 32000  # 32 samples per iteration * 1000 iterations


def test_training_start_end_without_single_iteration_callbacks(
    mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, config: TrainingTelemetryConfig
) -> None:
    """Test that training start/end callbacks create and stop the appropriate spans when we don't get callbacks for individual iterations."""
    # global step starts from 0. So this means that 10 iterations have been completed in a previous run.
    # So the first iteration of the current run is iteration # 10.
    train_iterations_start = 10
    train_samples_start = 0

    assert config.training_loop_config is not None
    config.training_loop_config.seq_length_or_fn = 1024
    reconfigure_provider(config, mock_exporter)

    on_train_start(
        train_iterations_start=train_iterations_start,
        train_samples_start=train_samples_start,
        train_iterations_target_or_fn=_dummy_train_iterations_target_fn,
        train_samples_target_or_fn=_dummy_train_samples_target_fn,
    )

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.TRAINING_LOOP
    assert span.attributes == TrainingLoopAttributes.create(
        perf_tag="test_perf",
        log_every_n_train_iterations=10,
        world_size=10,
        global_batch_size=32,
        train_iterations_start=train_iterations_start,
        train_samples_start=train_samples_start,
        train_iterations_target=1000,
        train_samples_target=32000,
        train_tokens_target=1024 * 32000,  # seq_length * train_samples_target
        completed_floating_point_operations_overall=train_iterations_start
        * 32
        * 100,  # 10 iterations in the loaded checkpoint * 32 samples per iteration * 100 flops per sample
        seq_length=1024,
    )

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_train_end()

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.TRAINING_LOOP
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 10000})

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            # No metrics update event is exported because we didn't get callbacks for individual iterations.
            Exporter.export_stop,
        ],
    )


def test_training_start_end_with_single_iteration_callbacks(
    mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, config: TrainingTelemetryConfig
) -> None:
    """Test that training start/end callbacks create and stop the appropriate spans when we get callbacks for individual iterations."""
    assert config.training_loop_config is not None
    config.training_loop_config.log_every_n_train_iterations = 8
    config.training_loop_config.flops_per_sample_or_fn = 100
    config.training_loop_config.global_batch_size_or_fn = 32
    config.training_loop_config.world_size_or_fn = 10
    config.training_loop_config.seq_length_or_fn = 1024
    reconfigure_provider(config, mock_exporter)

    expected_first_logged_train_iterations_finish_timestamp_sec = 0

    # global step starts from 0. So this means that 10 iterations have been completed in a previous run.
    # So the first iteration of the current run is iteration # 10.
    train_iterations_start = 10
    train_samples_start = 0

    on_train_start(
        train_iterations_start=train_iterations_start,
        train_samples_start=train_samples_start,
        train_iterations_target_or_fn=_dummy_train_iterations_target_fn,
        train_samples_target_or_fn=_dummy_train_samples_target_fn,
    )
    ts = advance_time(mock_time, mock_perf_counter, 10.0)

    for i in range(10):
        on_training_single_iteration_start()
        ts = advance_time(mock_time, mock_perf_counter, 50.0)
        on_training_single_iteration_end()
        if i == 0:
            expected_first_logged_train_iterations_finish_timestamp_sec = ts.seconds_since_epoch

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.TRAINING_LOOP
    assert span.attributes == TrainingLoopAttributes.create(
        perf_tag="test_perf",
        log_every_n_train_iterations=8,
        world_size=10,
        global_batch_size=32,
        train_iterations_start=train_iterations_start,
        train_samples_start=train_samples_start,
        train_iterations_target=1000,
        train_samples_target=32000,
        train_tokens_target=1024 * 32000,  # seq_length * train_samples_target
        completed_floating_point_operations_overall=train_iterations_start
        * 32
        * 100,  # 10 iterations in the loaded checkpoint * 32 samples per iteration * 100 flops per sample
        seq_length=1024,
    )

    on_train_end()
    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.TRAINING_LOOP
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: (10 + 50 * 10) * 1000})

    assert mock_exporter.export_event.call_count == 1
    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.TRAINING_METRICS_UPDATE
    # Make sure the event exported is the same as the event stored in the span.
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 1
    assert span_events[0] == event

    # We update the metrics every 8 iterations. Since we start from iteration 10, the event
    # is fired at iteration number 15 (because we add 1 to the iteration number before deciding
    # when to send the metrics), which means after completing 6 iterations.
    expected_ev_attributes = TrainingMetricsUpdateAttributes.create(
        train_iterations_start=train_iterations_start,
        current_iteration=15,
        num_iterations=6,
        train_samples_start=train_samples_start,
        num_train_samples=6 * 32,
        interval=8,
        avg_iteration_time_sec=50.0,
        min_iteration_time_sec=50.0,
        max_iteration_time_sec=50.0,
        total_iteration_time_sec=50.0 * 6,
        train_tokens=1024 * 6 * 32,
        # 10 iterations in the previous run (iterations 0 upto and incl 9) and 6 iterations in the current job.
        completed_floating_point_operations_overall=(10 + 6) * 32 * 100,
        total_flops=32 * 100 * 6,
        train_throughput_per_gpu=32 * 100.0 / (50.0 * 10**12 * 10),
        train_throughput_per_gpu_max=32 * 100.0 / (50.0 * 10**12 * 10),
        train_throughput_per_gpu_min=32 * 100.0 / (50.0 * 10**12 * 10),
        first_logged_train_iterations_finish_timestamp_sec=expected_first_logged_train_iterations_finish_timestamp_sec,
        # first_logged_train_iterations_finish_timestamp_sec was captured at the end of the first iteration and
        # last_logged_train_iterations_finish_timestamp_sec was captured at the end of the 6th iteration.
        last_logged_train_iterations_finish_timestamp_sec=expected_first_logged_train_iterations_finish_timestamp_sec + 50 * 5,
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10 + 50 * 6) * 1000)
    assert event.attributes == expected_ev_attributes

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_validation_loop_callbacks_without_single_iteration_callbacks(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that validation loop callbacks create and stop the appropriate spans when we don't get callbacks for individual iterations."""
    on_validation_start()

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert not span.attributes

    advance_time(mock_time, mock_perf_counter, 10.0)

    on_validation_end()

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 10 * 1000})

    assert mock_exporter.export_event.call_count == 1
    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE
    # Make sure the event exported is the same as the event stored in the span.
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 1
    assert span_events[0] == event

    expected_ev_attributes = ValidationMetricsUpdateAttributes.create(
        current_iteration=0,
        interval=0,
        # The below values are None because we didn't get callbacks for individual iterations.
        avg_iteration_time_sec=None,
        min_iteration_time_sec=None,
        max_iteration_time_sec=None,
        total_iteration_time_sec=None,
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10) * 1000)
    assert event.attributes == expected_ev_attributes

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_validation_loop_callbacks_with_single_iteration_callbacks(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that validation loop callbacks create and stop the appropriate spans when we get callbacks for individual iterations."""
    on_validation_start()

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert not span.attributes

    advance_time(mock_time, mock_perf_counter, 10.0)

    for _ in range(10):
        on_validation_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 50.0)
        on_validation_single_iteration_end()

    on_validation_end()

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: (10 + 10 * 50) * 1000})

    assert mock_exporter.export_event.call_count == 1
    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE
    # Make sure the event exported is the same as the event stored in the span.
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 1
    assert span_events[0] == event

    expected_ev_attributes = ValidationMetricsUpdateAttributes.create(
        current_iteration=0,
        interval=0,
        avg_iteration_time_sec=50.0,
        min_iteration_time_sec=50.0,
        max_iteration_time_sec=50.0,
        total_iteration_time_sec=50.0 * 10,
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10 + 10 * 50) * 1000)
    assert event.attributes == expected_ev_attributes

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_testing_loop_callbacks(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that testing loop callbacks create and stop the appropriate spans."""
    on_testing_start()

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.TESTING_LOOP
    assert not span.attributes

    advance_time(mock_time, mock_perf_counter, 10.0)

    on_testing_end()

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.TESTING_LOOP
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 10000})

    assert mock_exporter.export_event.call_count == 1
    event = event_from_export_event(mock_exporter, span)
    assert event.name == StandardTrainingJobEventName.TESTING_METRICS_UPDATE
    # Make sure the event exported is the same as the event stored in the span.
    span_events = get_non_trivial_events(span)
    assert len(span_events) == 1
    assert span_events[0] == event

    expected_ev_attributes = TestingMetricsUpdateAttributes.create(current_iteration=0, interval=0)
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 10) * 1000)
    assert event.attributes == expected_ev_attributes

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_invalid_callback_usage(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that invalid callback usage is handled correctly.

    This test confirms our assumption that for standard training spans, we can only start a span once and must end it before
    starting a new span of the same type.
    It also confirms that we cannot end a span that was not started.
    """
    on_training_single_iteration_start()
    with pytest.raises(OneLoggerError, match="Cannot start timer since it is already active"):
        on_training_single_iteration_start()

    with pytest.raises(OneLoggerError, match="Expected to have one and only one span named model_initialization but found 0."):
        on_model_init_end()
        on_model_init_end()


@pytest.mark.parametrize(
    "start_time_msec,finish_time_msec",
    [
        (None, None),
        ((STARTING_TIME - 40) * 1000, None),
        ((STARTING_TIME - 40) * 1000, (STARTING_TIME - 2) * 1000),
    ],
    ids=["no_times", "start_time_only", "both_times"],
)
def test_load_checkpoint_callbacks(
    mock_exporter: MagicMock,
    mock_perf_counter: Mock,
    mock_time: Mock,
    start_time_msec: Union[float, None],
    finish_time_msec: Union[float, None],
) -> None:
    """Test that load checkpoint callbacks create and stop the appropriate spans.

    Args:
        mock_exporter: Mocked exporter instance
        mock_perf_counter: Mocked perf counter
        mock_time: Mocked time
        start_time_msec: Optional start time in milliseconds. If None, current time will be used.
        finish_time_msec: Optional end time in milliseconds. If None, current time will be used.
    """
    on_load_checkpoint_start(start_time_msec)

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_LOAD
    assert span.attributes == Attributes({})

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_load_checkpoint_end(finish_time_msec)

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_LOAD

    expected_start = start_time_msec if start_time_msec is not None else STARTING_TIME * 1000
    expected_finish = finish_time_msec if finish_time_msec is not None else (STARTING_TIME + 10) * 1000
    expected_duration = expected_finish - expected_start

    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: expected_duration})

    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


@pytest.mark.parametrize(
    "start_time_msec,finish_time_msec",
    [
        (None, None),
        ((STARTING_TIME - 40) * 1000, None),
        ((STARTING_TIME - 40) * 1000, (STARTING_TIME - 2) * 1000),
    ],
    ids=["no_times", "start_time_only", "both_times"],
)
def test_optimizer_init_callbacks(
    mock_exporter: MagicMock,
    mock_perf_counter: Mock,
    mock_time: Mock,
    start_time_msec: Union[float, None],
    finish_time_msec: Union[float, None],
) -> None:
    """Test that optimizer initialization callbacks create and stop the appropriate spans.

    Args:
        mock_exporter: Mocked exporter instance
        mock_perf_counter: Mocked perf counter
        mock_time: Mocked time
        start_time_msec: Optional start time in milliseconds. If None, current time will be used.
        finish_time_msec: Optional end time in milliseconds. If None, current time will be used.
    """
    on_optimizer_init_start(start_time_msec)

    # Verify span start
    assert mock_exporter.export_start.call_count == 1
    span = span_from_export_start(mock_exporter, None)
    assert span.name == StandardTrainingJobSpanName.OPTIMIZER_INIT
    assert span.attributes == Attributes({})

    advance_time(mock_time, mock_perf_counter, 10.0)
    on_optimizer_init_end(finish_time_msec)

    # Verify span end
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.OPTIMIZER_INIT

    expected_start = start_time_msec if start_time_msec is not None else STARTING_TIME * 1000
    expected_finish = finish_time_msec if finish_time_msec is not None else (STARTING_TIME + 10) * 1000
    expected_duration = expected_finish - expected_start

    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: expected_duration})

    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_disabled_for_current_rank(config: TrainingTelemetryConfig, mock_exporter: MagicMock) -> None:
    """Test that the training telemetry is disabled for the current rank."""
    config.enable_for_current_rank = False
    reset_singletong_providers_for_test()
    (TrainingTelemetryProvider.instance().with_base_telemetry_config(config).with_exporter(mock_exporter).configure_provider())

    # Try a few callbacks to make sure the provider is disabled.
    assert on_app_start() is None
    assert on_model_init_start() is None
    assert on_model_init_end() is None
    assert on_dataloader_init_start() is None
    assert on_dataloader_init_end() is None
    assert on_load_checkpoint_start() is None
    assert on_load_checkpoint_end() is None
    assert on_optimizer_init_start() is None
    assert on_optimizer_init_end() is None
    assert on_train_start() is None
    assert on_train_end() is None
    assert on_validation_start() is None
    assert on_validation_end() is None
    assert on_testing_start() is None
    assert on_testing_end() is None
    assert on_save_checkpoint_start() is None
    assert on_save_checkpoint_success() is None
    assert on_save_checkpoint_end() is None
    assert on_app_end() is None

    mock_exporter.assert_not_called()

    # Undo the force disable logging so that other tests don't fail.
    OneLoggerProvider.instance()._logging_force_disabled = False
