# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the training telemetry API.

Tests in this module are integration tests that test the full lifecycle of a typical training job.
These tests use the training telemetry API with a mock exporter to verify what gets exported for a training job.
"""
# pyright: reportPrivateUsage=false
import socket
from typing import Any, Callable, List
from unittest.mock import MagicMock, Mock

import pytest
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import StandardEventAttributeName
from nv_one_logger.core.span import StandardSpanAttributeName, StandardSpanName
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

from .conftest import configure_provider_for_test
from .utils import (
    advance_time,
    assert_exporter_method_call_sequence,
    assert_no_export,
    assert_only_start_event,
    assert_only_start_stop_event,
    event_from_export_event,
    get_non_trivial_events,
    span_from_export_start,
    span_from_export_stop,
)


@pytest.fixture(autouse=True)
def configure_provider(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
    """Fixture that configures the TrainingTelemetryProvider."""
    configure_provider_for_test(config, mock_exporter)


STARTING_PERF_COUNTER = 200
STARTING_TIME = 5000


def _cur_ts(mock_time: Mock) -> int:
    return int(mock_time.return_value * 1000)


def test_training_e2e(config: TrainingTelemetryConfig, mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Tests the full lifecycle of a typical training job."""
    mock_time.return_value = STARTING_TIME
    mock_perf_counter.return_value = STARTING_PERF_COUNTER

    config.log_every_n_train_iterations = 2
    config.seq_length_or_fn = 1024

    train_iterations_start = 100
    num_train_iterations = 11
    checkpoint_interval = 3
    validation_interval = 5
    num_validation_iterations = 4

    # Each element is a tuple of (method_name, list of args, dict of kwargs)
    expected_calls: List[Callable[..., Any]] = [Exporter.initialize]
    on_app_start()
    app_span = span_from_export_start(mock_exporter, None)
    assert app_span.name == StandardSpanName.APPLICATION
    assert app_span.attributes == Attributes({})
    assert app_span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: 5000000})
    event = event_from_export_event(mock_exporter, app_span)
    assert event.name == StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION
    assert event.attributes == OneLoggerInitializationAttributes.create(
        one_logger_training_telemetry_version="2.0.0",
        enable_for_current_rank=config.enable_for_current_rank,
        perf_tag=config.perf_tag,
        session_tag=config.session_tag,
        app_type=config.app_type,
        log_every_n_train_iterations=config.log_every_n_train_iterations,
        world_size=config.world_size,
        global_batch_size=config.global_batch_size,
        is_baseline_run=config.is_baseline_run,
        is_train_iterations_enabled=config.is_train_iterations_enabled,
        is_validation_iterations_enabled=config.is_validation_iterations_enabled,
        is_test_iterations_enabled=config.is_test_iterations_enabled,
        is_save_checkpoint_enabled=config.is_save_checkpoint_enabled,
        is_log_throughput_enabled=config.is_log_throughput_enabled,
        node_name=socket.gethostname(),
        rank=0,
        summary_data_schema_version=config.summary_data_schema_version,
        checkpoint_strategy=CheckPointStrategy.SYNC,
        seq_length=config.seq_length,
    ).add(StandardEventAttributeName.TIMESTAMP_MSEC, 5000000)

    expected_calls.append(Exporter.export_start)
    expected_calls.append(Exporter.export_event)  # For ONE_LOGGER_INITIALIZATION event
    advance_time(mock_time, mock_perf_counter, 10.0)  # move perf counter to 5010

    # DATA_LOADER_INIT span
    on_dataloader_init_start()
    expected_calls.append(Exporter.export_start)
    span = span_from_export_start(mock_exporter, app_span)
    assert span.name == StandardTrainingJobSpanName.DATA_LOADER_INIT
    assert_only_start_event(span)
    assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    advance_time(mock_time, mock_perf_counter, 20.0)  # move perf counter to 5030
    on_dataloader_init_end()
    expected_calls.append(Exporter.export_stop)
    assert span == span_from_export_stop(mock_exporter)
    assert_only_start_stop_event(span, mock_exporter)
    assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    # CHECKPOINT_LOAD span
    advance_time(mock_time, mock_perf_counter, 20.5)  # move perf counter to 5050.5
    on_load_checkpoint_start()
    expected_calls.append(Exporter.export_start)
    span = span_from_export_start(mock_exporter, app_span)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_LOAD
    assert span.attributes == Attributes({})
    assert_only_start_event(span)
    assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    advance_time(mock_time, mock_perf_counter, 9.5)  # move perf counter to 5060
    on_load_checkpoint_end()
    expected_calls.append(Exporter.export_stop)
    assert span == span_from_export_stop(mock_exporter)
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 9500})
    assert_only_start_stop_event(span, mock_exporter)
    assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    # MODEL_INIT span
    advance_time(mock_time, mock_perf_counter, 2.0)  # move perf counter to 5062
    on_model_init_start()
    expected_calls.append(Exporter.export_start)
    span = span_from_export_start(mock_exporter, app_span)
    assert span.name == StandardTrainingJobSpanName.MODEL_INIT
    assert span.attributes == Attributes({})
    assert_only_start_event(span)
    assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    advance_time(mock_time, mock_perf_counter, 68.0)  # move perf counter to 5130
    on_model_init_end()
    expected_calls.append(Exporter.export_stop)
    assert span == span_from_export_stop(mock_exporter)
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 68000})
    assert_only_start_stop_event(span, mock_exporter)
    assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    # OPTIMIZER_INIT span
    on_optimizer_init_start()
    expected_calls.append(Exporter.export_start)
    span = span_from_export_start(mock_exporter, app_span)
    assert span.name == StandardTrainingJobSpanName.OPTIMIZER_INIT
    assert span.attributes == Attributes({})
    assert_only_start_event(span)
    assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    advance_time(mock_time, mock_perf_counter, 10.0)  # move perf counter to 5140
    on_optimizer_init_end()
    expected_calls.append(Exporter.export_stop)
    assert span == span_from_export_stop(mock_exporter)
    assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 10000})
    assert_only_start_stop_event(span, mock_exporter)
    assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    # TRAINING_LOOP span
    training_loop_start = advance_time(mock_time, mock_perf_counter, 10.0)  # move perf counter to 5150
    on_train_start(
        train_iterations_start=train_iterations_start,
        train_samples_start=3200,
        train_iterations_target_or_fn=1000,
        train_samples_target_or_fn=32000,
    )
    expected_calls.append(Exporter.export_start)
    span = span_from_export_start(mock_exporter, app_span)
    training_loop_span = span
    assert span.name == StandardTrainingJobSpanName.TRAINING_LOOP
    assert span.attributes == TrainingLoopAttributes.create(
        train_iterations_start=train_iterations_start,
        train_iterations_target=1000,
        train_samples_start=3200,
        train_samples_target=32000,
        train_tokens_target=32000 * 1024,  # train_samples_target * seq_length
        completed_floating_point_operations_overall=320000,  # 10 iterations in the loaded checkpoint * 32 samples per iteration * 100 flops per sample
    )
    assert_only_start_event(span)
    assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    advance_time(mock_time, mock_perf_counter, 50.0)  # move perf counter to 5200

    expected_total_iteration_time_sec = 0.0
    expected_train_throughput_per_gpu_min = float("inf")
    expected_train_throughput_per_gpu_max = 0.0

    expected_saved_checkpoints_so_far = 0
    expected_validation_loops_so_far = 0
    expected_first_ckpt_ts = None
    expected_first_iter_finish_ts = None
    expected_validation_time_sec = 0.0
    # TRAINING_SINGLE_ITERATION span
    # iteration is the iteration number relative to the start of this training job.
    for iteration in range(0, num_train_iterations):
        # cur_iteration is the absolute iteration number
        cur_iteration = iteration + train_iterations_start
        # We don't export spans of type TRAINING_SINGLE_ITERATION
        with assert_no_export(mock_exporter):
            on_training_single_iteration_start()

        # Introduce some variation in the iteration time.
        if iteration % 2 == 0:
            advance_time(mock_time, mock_perf_counter, 400.0)
            expected_total_iteration_time_sec += 400.0
        else:
            advance_time(mock_time, mock_perf_counter, 500.0)
            expected_total_iteration_time_sec += 500.0
        expected_train_throughput_per_gpu = (
            config.global_batch_size * config.flops_per_sample * (iteration + 1) / (expected_total_iteration_time_sec * 10**12 * config.world_size)
        )
        expected_train_throughput_per_gpu_min = min(expected_train_throughput_per_gpu_min, expected_train_throughput_per_gpu)
        expected_train_throughput_per_gpu_max = max(expected_train_throughput_per_gpu_max, expected_train_throughput_per_gpu)
        prev_export_count = len(mock_exporter.mock_calls)
        if not expected_first_iter_finish_ts:
            expected_first_iter_finish_ts = TracingTimestamp.now()
        on_training_single_iteration_end()
        latest_iteration_finish_ts = TracingTimestamp.now()

        # We send out metrics every N iterations where n=log_every_n_train_iterations
        if cur_iteration > 0 and (cur_iteration + 1) % config.log_every_n_train_iterations == 0:
            expected_calls.append(Exporter.export_event)  # For TRAINING_METRICS_UPDATE event
            event = event_from_export_event(mock_exporter, training_loop_span)
            assert event.name == StandardTrainingJobEventName.TRAINING_METRICS_UPDATE
            # If iteration is N and cur_iteration is M, it means we just finished the iteration number M,
            # which means we have completed N+1 iterations in this job and M + 1 iterations overall.

            assert event.attributes == TrainingMetricsUpdateAttributes.create(
                train_iterations_start=train_iterations_start,
                current_iteration=cur_iteration,
                num_iterations=iteration + 1,
                train_samples_start=3200,
                num_train_samples=(iteration + 1) * config.global_batch_size,
                interval=config.log_every_n_train_iterations,
                avg_iteration_time_sec=(expected_total_iteration_time_sec / (iteration + 1)),
                min_iteration_time_sec=400.0,
                max_iteration_time_sec=500.0 if iteration > 0 else 400.0,
                total_iteration_time_sec=expected_total_iteration_time_sec,
                train_tokens=config.global_batch_size * config.seq_length * (iteration + 1),
                completed_floating_point_operations_overall=(cur_iteration + 1) * config.global_batch_size * config.flops_per_sample,
                total_flops=config.global_batch_size * config.flops_per_sample * (iteration + 1),
                train_throughput_per_gpu=expected_train_throughput_per_gpu,
                train_throughput_per_gpu_max=expected_train_throughput_per_gpu_max,
                train_throughput_per_gpu_min=expected_train_throughput_per_gpu_min,
                first_logged_train_iterations_finish_timestamp_sec=expected_first_iter_finish_ts.seconds_since_epoch,
                last_logged_train_iterations_finish_timestamp_sec=latest_iteration_finish_ts.seconds_since_epoch,
            ).add(StandardEventAttributeName.TIMESTAMP_MSEC, _cur_ts(mock_time))
        else:
            assert len(mock_exporter.mock_calls) == prev_export_count, "Exporter was called unnecessarily for TRAINING_SINGLE_ITERATION"

        if iteration % checkpoint_interval == 0:
            # CHECKPOINT_SAVE_SYNC span
            on_save_checkpoint_start(global_step=cur_iteration)
            expected_saved_checkpoints_so_far += 1
            expected_calls.append(Exporter.export_start)
            span = span_from_export_start(mock_exporter, training_loop_span)
            assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
            assert span.attributes == CheckpointSaveSpanAttributes.create(
                checkpoint_strategy=CheckPointStrategy.SYNC, current_iteration=cur_iteration, save_checkpoint_attempt_count=expected_saved_checkpoints_so_far
            )
            assert_only_start_event(span)
            assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

            expected_latest_ckpt_ts = advance_time(mock_time, mock_perf_counter, 100.0)
            if not expected_first_ckpt_ts:
                expected_first_ckpt_ts = expected_latest_ckpt_ts
            on_save_checkpoint_success(global_step=cur_iteration)
            expected_calls.append(Exporter.export_event)  # For SAVE_CHECKPOINT_SUCCESS event
            event = event_from_export_event(mock_exporter, span)
            assert event.name == StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS
            expected_ev_attributes: Attributes = SaveCheckpointSuccessEventAttributes.create(
                checkpoint_strategy=CheckPointStrategy.SYNC,
                current_iteration=cur_iteration,
                first_successful_save_checkpoint_timestamp_sec=expected_first_ckpt_ts.seconds_since_epoch,
                latest_successful_save_checkpoint_timestamp_sec=expected_latest_ckpt_ts.seconds_since_epoch,
                save_checkpoint_success_count=expected_saved_checkpoints_so_far,
                training_start_timestamp_sec=training_loop_start.seconds_since_epoch,
                productive_train_iterations=cur_iteration + 1,
                productive_train_samples=config.global_batch_size * (cur_iteration + 1),
                productive_train_iterations_sec=expected_total_iteration_time_sec,
                productive_validation_iterations_sec=expected_validation_time_sec,
                productive_train_tflops=config.global_batch_size * config.flops_per_sample * (cur_iteration + 1) / 10**12,
            )
            expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, _cur_ts(mock_time))
            assert event.attributes == expected_ev_attributes

            advance_time(mock_time, mock_perf_counter, 10.0)
            on_save_checkpoint_end()
            expected_calls.append(Exporter.export_event)  # For SYNC_CHECKPOINT_METRICS_UPDATE event.
            expected_calls.append(Exporter.export_stop)
            assert span == span_from_export_stop(mock_exporter)
            assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 110000})
            # start, stop, SAVE_CHECKPOINT_SUCCESS event and SYNC_CHECKPOINT_METRICS_UPDATE event
            assert len(span.events) == 4

            event = get_non_trivial_events(span)[1]
            assert event.name == StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE
            expected_ev_attributes = SyncCheckpointMetricsUpdateAttributes.create(
                save_checkpoint_sync_time_total_sec=110.0 * expected_saved_checkpoints_so_far,
                save_checkpoint_sync_time_min_sec=110.0,
                save_checkpoint_sync_time_max_sec=110.0,
            )
            expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, _cur_ts(mock_time))
            assert event.attributes == expected_ev_attributes

            assert span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

        if iteration % validation_interval == 0:
            # VALIDATION_LOOP span
            on_validation_start()
            expected_validation_loops_so_far += 1
            expected_calls.append(Exporter.export_start)
            span = span_from_export_start(mock_exporter, training_loop_span)
            assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
            assert span.attributes == Attributes({})
            assert_only_start_event(span)
            assert span.start_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

            for _ in range(num_validation_iterations):
                # VALIDATION_SINGLE_ITERATION span
                # We don't export spans of type VALIDATION_SINGLE_ITERATION
                with assert_no_export(mock_exporter):
                    on_validation_single_iteration_start()

                advance_time(mock_time, mock_perf_counter, 200.0)
                expected_validation_time_sec += 200.0
                with assert_no_export(mock_exporter):
                    on_validation_single_iteration_end()

                advance_time(mock_time, mock_perf_counter, 5)

            on_validation_end()
            expected_calls.append(Exporter.export_event)  # For VALIDATION_METRICS_UPDATE event
            expected_calls.append(Exporter.export_stop)
            assert span == span_from_export_stop(mock_exporter)
            assert span.updated_attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: (200 + 5) * num_validation_iterations * 1000})

            event = event_from_export_event(mock_exporter, span)
            # Make sure the event exported is the same as the event stored in the span.
            span_events = get_non_trivial_events(span)
            assert len(span_events) == 1
            assert span_events[0] == event
            assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE

            # The training loop in this test starts a validation loop every 5 iterations. But since
            # in this test, we start from iteration 100, for the first validation loop, the interval
            # is only 1 instead of 5.
            expected_interval = validation_interval if iteration > 0 else 1
            expected_val_ev_attributes = ValidationMetricsUpdateAttributes.create(
                current_iteration=cur_iteration,
                interval=expected_interval,
                avg_iteration_time_sec=200.0,
                min_iteration_time_sec=200.0,
                max_iteration_time_sec=200.0,
                total_iteration_time_sec=200.0 * num_validation_iterations * expected_validation_loops_so_far,
            )
            expected_val_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, _cur_ts(mock_time))
            assert event.attributes == expected_val_ev_attributes

            advance_time(mock_time, mock_perf_counter, 5.0)
        advance_time(mock_time, mock_perf_counter, 400.0)

    on_train_end()
    expected_calls.append(Exporter.export_stop)
    assert training_loop_span == span_from_export_stop(mock_exporter)
    assert training_loop_span.updated_attributes == Attributes(
        {StandardSpanAttributeName.DURATION_MSEC: int(mock_perf_counter.return_value - training_loop_start.perf_counter_seconds) * 1000}
    )
    # start, stop and 5 TRAINING_METRICS_UPDATE events
    assert len(training_loop_span.events) == 7
    assert training_loop_span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    advance_time(mock_time, mock_perf_counter, 500.0)
    on_app_end()
    expected_calls.append(Exporter.export_stop)
    assert app_span == span_from_export_stop(mock_exporter)
    assert app_span.updated_attributes == Attributes(
        {StandardSpanAttributeName.DURATION_MSEC: int(mock_perf_counter.return_value - STARTING_PERF_COUNTER) * 1000}
    )
    # start, stop and 1 TRAINING_METRICS_UPDATE event
    assert len(app_span.events) == 3
    assert app_span.stop_event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: _cur_ts(mock_time)})

    # make sure we exported extactly what we expected.
    expected_calls.append(Exporter.close)
    assert_exporter_method_call_sequence(mock_exporter, expected_calls)
    assert_exporter_method_call_sequence(mock_exporter, expected_calls)


def test_training_e2e_disabled_for_current_rank(config: TrainingTelemetryConfig, mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Tests the full lifecycle of a typical training job with training telemetry disabled for the current rank."""
    config.enable_for_current_rank = False
    configure_provider_for_test(config, mock_exporter)

    mock_time.return_value = STARTING_TIME
    mock_perf_counter.return_value = STARTING_PERF_COUNTER

    assert on_app_start() is None

    # DATA_LOADER_INIT span
    assert on_dataloader_init_start() is None
    assert on_dataloader_init_end() is None

    # CHECKPOINT_LOAD span
    assert on_load_checkpoint_start() is None
    assert on_load_checkpoint_end() is None

    # MODEL_INIT span
    assert on_model_init_start() is None
    assert on_model_init_end() is None

    # OPTIMIZER_INIT span
    assert on_optimizer_init_start() is None
    assert on_optimizer_init_end() is None

    # TRAINING_LOOP span
    assert (
        on_train_start(
            train_iterations_start=0,
            train_samples_start=3200,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=32000,
        )
        is None
    )
    # TRAINING_SINGLE_ITERATION span
    for cur_iteration in range(0, 20):
        assert on_training_single_iteration_start() is None
        assert on_training_single_iteration_end() is None
        if cur_iteration % 2 == 0:
            assert on_save_checkpoint_start(global_step=cur_iteration) is None
            assert on_save_checkpoint_success(global_step=cur_iteration) is None
            assert on_save_checkpoint_end() is None
        if cur_iteration % 2 == 0:
            # VALIDATION_LOOP span
            assert on_validation_start() is None
            for _ in range(3):
                # VALIDATION_SINGLE_ITERATION span
                assert on_validation_single_iteration_start() is None
                assert on_validation_single_iteration_end() is None
            assert on_validation_end() is None

    assert on_train_end() is None
    assert on_app_end() is None

    mock_exporter.assert_not_called()

    # Undo the force disable logging so that other tests don't fail.
    OneLoggerProvider.instance()._logging_force_disabled = False
