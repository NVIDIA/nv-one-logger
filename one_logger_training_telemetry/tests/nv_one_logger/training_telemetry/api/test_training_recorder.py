# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TrainingRecorder class."""
# pyright: reportPrivateUsage=false

from typing import Set
from unittest.mock import Mock

import pytest
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.metric_summarizer import MetricSummarizer
from nv_one_logger.core.internal.multi_window_timer import MultiWindowTimer
from nv_one_logger.core.span import Span, SpanName, StandardSpanName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.recorder.default_recorder import ExportCustomizationMode

from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder, _TrainingState
from nv_one_logger.training_telemetry.api.training_telemetry_provider import DEFAULT_SPANS_EXPORT_BLACKLIST

from .utils import advance_time

STARTING_PERF_COUNTER = 5000.0
STARTING_TIME = 120000.0


@pytest.fixture(autouse=True)
def initialize_time(mock_time: Mock, mock_perf_counter: Mock) -> None:
    """Initialize the time and perf counter mocks."""
    mock_time.return_value = STARTING_TIME
    mock_perf_counter.return_value = STARTING_PERF_COUNTER


@pytest.fixture
def training_recorder(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> TrainingRecorder:
    """Create a TrainingRecorder instance for testing.

    Args:
        mock_exporter: A mock exporter instance.

    Returns:
        TrainingRecorder: A configured TrainingRecorder instance.
    """
    recorder = TrainingRecorder(
        config=config,
        exporters=[mock_exporter],
        export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS,
        span_name_filter=DEFAULT_SPANS_EXPORT_BLACKLIST,
    )
    OneLoggerProvider.instance()._config = config
    OneLoggerProvider.instance()._recorder = recorder

    return recorder


def test_training_recorder_initialization(training_recorder: TrainingRecorder) -> None:
    """Test that TrainingRecorder initializes correctly.

    This test verifies that:
    1. The TrainingRecorder is properly initialized with exporters
    2. The training state is initialized with default values
    3. The multi-iteration timers are properly set up

    Args:
        training_recorder: A configured TrainingRecorder instance.
    """
    # Access protected members for testing purposes
    state = training_recorder._training_state
    assert isinstance(state, _TrainingState)
    assert state.completed_training_iterations_overall == 0
    assert state.completed_floating_point_operations_overall == 0
    assert state.total_flops_current_job == 0
    assert state.train_samples_start_processed_current_job == 0
    assert state.first_logged_train_iterations_finish_time is None
    assert state.first_save_checkpoint_success_time is None

    # Check that all required timers are initialized
    expected_timers: Set[SpanName] = {
        StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION,
        StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION,
        StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC,
        StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC,
        StandardTrainingJobSpanName.CHECKPOINT_LOAD,
    }
    assert set(state.multi_iteration_timers.keys()) == expected_timers


def test_active_spans(training_recorder: TrainingRecorder) -> None:
    """Tests that the list of active spans is updated correctly."""
    span1 = training_recorder.on_app_start(start_time=TracingTimestamp.now())
    assert len(training_recorder._spans) == 1
    assert training_recorder._get_active_span(StandardSpanName.APPLICATION) == span1

    span2 = training_recorder.on_dataloader_init_start(start_time=TracingTimestamp.now())
    assert len(training_recorder._spans) == 2
    assert training_recorder.get_active_spans_by_name(StandardSpanName.APPLICATION) == [span1]
    assert training_recorder.get_active_spans_by_name(StandardTrainingJobSpanName.DATA_LOADER_INIT) == [span2]

    training_recorder.stop(span2)
    assert len(training_recorder._spans) == 1
    assert training_recorder.get_active_spans_by_name(StandardSpanName.APPLICATION) == [span1]

    training_recorder.stop(span1)
    assert len(training_recorder._spans) == 0


def test_training_state(training_recorder: TrainingRecorder, config: TrainingTelemetryConfig, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Tests that the training state is updated correctly.

    This test verifies that:
    1. Training state is updated correctly after single iteration
    2. State accumulates correctly over multiple iterations
    3. Checkpoint save events update the state correctly
    4. Multiple checkpoint saves update timestamps correctly
    """
    train_iterations_start = 5
    training_recorder.on_training_loop_start(train_iterations_start=train_iterations_start, train_samples_start=3000)

    expected_state = _TrainingState(
        multi_iteration_timers={
            StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION: MultiWindowTimer(),
            StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION: MultiWindowTimer(),
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC: MultiWindowTimer(),
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC: MultiWindowTimer(),
            StandardTrainingJobSpanName.CHECKPOINT_LOAD: MultiWindowTimer(),
        },
        train_iterations_start=train_iterations_start,
        completed_training_iterations_overall=5,  # We will start from iteration 5 so we have completed iterations 0-4 (5 iterations in total).
        train_samples_start=3000,
        train_samples_start_processed_current_job=0,
        total_flops_current_job=0,
        train_tokens_current_job=None,
        completed_floating_point_operations_overall=train_iterations_start * config.global_batch_size * config.flops_per_sample,
        validation_interval_start=5,
        testing_interval_start=5,
        first_logged_train_iterations_finish_time=None,
        last_logged_train_iterations_finish_time=None,
        first_save_checkpoint_success_time=None,
        latest_save_checkpoint_success_time=None,
        tflops_per_gpu=MetricSummarizer[float](),
    )

    assert training_recorder._training_state == expected_state

    # ##########################################################
    # Load a checkpoint
    # ##########################################################
    latest_ts: TracingTimestamp = advance_time(mock_time, mock_perf_counter, 10.0)  # current time is 45010.0
    timer = MultiWindowTimer()
    timer.start()
    expected_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_LOAD] = timer
    training_recorder.on_load_checkpoint_start(start_time=latest_ts)

    latest_ts = advance_time(mock_time, mock_perf_counter, 10.0)  # current time is 45010.0
    training_recorder.on_load_checkpoint_end(stop_time=latest_ts)
    timer.stop()

    assert training_recorder._training_state == expected_state

    # ##########################################################
    # First training iteration
    # ##########################################################
    latest_ts = advance_time(mock_time, mock_perf_counter, 5.55)  # current time is 45015.55
    timer = MultiWindowTimer()
    timer.start()
    expected_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION] = timer
    training_recorder.on_training_single_iteration_start(start_time=latest_ts)

    assert training_recorder._training_state == expected_state

    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45025.55
    training_recorder.on_training_single_iteration_end(stop_time=latest_ts)
    timer.stop()

    expected_state.completed_training_iterations_overall = 6
    expected_state.completed_floating_point_operations_overall = 19200  # 6 iterations * 32 batch size * 100 flops
    expected_state.total_flops_current_job = 3200  # 32 batch size * 100 flops
    expected_state.train_samples_start_processed_current_job = 32  # batch size
    expected_state.first_logged_train_iterations_finish_time = latest_ts
    expected_state.last_logged_train_iterations_finish_time = latest_ts
    expected_state.tflops_per_gpu.add_value(3200 / (10 * 10**12 * config.world_size))
    assert training_recorder._training_state == expected_state

    # ##########################################################
    # Second training iteration
    # ##########################################################
    latest_ts = advance_time(mock_time, mock_perf_counter, 14.45)  # current time is 45030
    training_recorder.on_training_single_iteration_start(start_time=latest_ts)
    timer.start()

    latest_ts = advance_time(mock_time, mock_perf_counter, 20)  # current time is 45050
    training_recorder.on_training_single_iteration_end(stop_time=latest_ts)
    timer.stop()

    expected_state.completed_training_iterations_overall = 7
    expected_state.completed_floating_point_operations_overall = 22400  # 7 iterations * 32 batch size * 100 flops
    expected_state.total_flops_current_job = 3200 * 2
    expected_state.train_samples_start_processed_current_job = 32 * 2
    expected_state.last_logged_train_iterations_finish_time = latest_ts
    expected_state.tflops_per_gpu.add_value(3200 * 2 / ((10 + 20) * 10**12 * config.world_size))
    assert training_recorder._training_state == expected_state

    # ##########################################################
    # First checkpoint save
    # ##########################################################
    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45040
    timer = MultiWindowTimer()
    timer.start()
    expected_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC] = timer
    training_recorder.on_save_checkpoint_start(current_iteration=1, start_time=latest_ts)

    latest_ts = advance_time(mock_time, mock_perf_counter, 20)  # current time is 45060
    event_time = latest_ts
    training_recorder.on_save_checkpoint_success(current_iteration=1, timestamp=latest_ts)
    training_recorder.on_save_checkpoint_end(stop_time=latest_ts)
    timer.stop()

    expected_state.first_save_checkpoint_success_time = event_time
    expected_state.latest_save_checkpoint_success_time = event_time
    expected_state.successful_save_checkpoint_count_current_job = 1

    assert training_recorder._training_state == expected_state

    # ##########################################################
    # First validation loop
    # ##########################################################
    latest_ts = advance_time(mock_time, mock_perf_counter, 30)  # current time is 45100
    training_recorder.on_validation_start(start_time=latest_ts)

    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45110
    timer = MultiWindowTimer()
    timer.start()
    expected_state.multi_iteration_timers[StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION] = timer
    training_recorder.on_validation_single_iteration_start(start_time=latest_ts)

    latest_ts = advance_time(mock_time, mock_perf_counter, 20)  # current time is 45130
    training_recorder.on_validation_single_iteration_end(stop_time=latest_ts)
    timer.stop()

    # another validation iteration
    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45140
    training_recorder.on_validation_single_iteration_start(start_time=latest_ts)
    timer.start()

    latest_ts = advance_time(mock_time, mock_perf_counter, 50)  # current time is 45190
    training_recorder.on_validation_single_iteration_end(stop_time=latest_ts)
    training_recorder.on_validation_end(stop_time=latest_ts)
    timer.stop()

    expected_state.validation_interval_start = 7
    assert training_recorder._training_state == expected_state

    # ##########################################################
    # Third training iteration
    # ##########################################################
    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45200
    timer = expected_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
    timer.start()
    training_recorder.on_training_single_iteration_start(start_time=latest_ts)

    latest_ts = advance_time(mock_time, mock_perf_counter, 20)  # current time is 45220
    training_recorder.on_training_single_iteration_end(stop_time=latest_ts)
    timer.stop()

    expected_state.completed_training_iterations_overall = 8
    expected_state.completed_floating_point_operations_overall = 25600  # 8 iterations * 32 batch size * 100 flops
    expected_state.total_flops_current_job = 3200 * 3
    expected_state.train_samples_start_processed_current_job = 32 * 3
    expected_state.last_logged_train_iterations_finish_time = latest_ts
    expected_state.tflops_per_gpu.add_value(3200 * 3 / ((10 + 20 + 20) * 10**12 * config.world_size))
    assert training_recorder._training_state == expected_state

    # ##########################################################
    # Second checkpoint save
    # ##########################################################
    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45230
    timer = expected_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC]
    training_recorder.on_save_checkpoint_start(current_iteration=2, start_time=latest_ts)
    timer.start()

    latest_ts = advance_time(mock_time, mock_perf_counter, 20)  # current time is 45250
    event_time = latest_ts
    training_recorder.on_save_checkpoint_success(current_iteration=1, timestamp=latest_ts)

    latest_ts = advance_time(mock_time, mock_perf_counter, 10)  # current time is 45260
    training_recorder.on_save_checkpoint_end(stop_time=latest_ts)
    timer.stop()

    expected_state.latest_save_checkpoint_success_time = event_time
    expected_state.successful_save_checkpoint_count_current_job = 2
    assert training_recorder._training_state == expected_state


def test_span_with_same_name_active(training_recorder: TrainingRecorder, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that creating a span with the same name as an active span raises an error.

    This test verifies that:
    1. Starting a span with a name that already has an active span raises an error
    2. The error message indicates that a span with the same name is already active
    """
    # Start first span
    span1 = training_recorder.on_training_single_iteration_start(TracingTimestamp.now())
    assert isinstance(span1, Span)

    # Attempting to start another span with the same name should raise an error
    with pytest.raises(OneLoggerError, match="already active"):
        training_recorder.on_training_single_iteration_start(advance_time(mock_time, mock_perf_counter, 10.0))


def test_events_for_typical_training_job(
    config: TrainingTelemetryConfig, training_recorder: TrainingRecorder, mock_perf_counter: Mock, mock_time: Mock
) -> None:
    """Tests that the correct metrics events are reported for a typical training job."""
    assert config.log_every_n_train_iterations == 10
    num_train_iterations = 90
    checkpoint_interval = 3
    validation_interval = 5

    # The spans are organized in the following way:
    #
    # APPLICATION
    #   DATA_LOADER_INIT
    #   CHECKPOINT_LOAD
    #   MODEL_INIT
    #   OPTIMIZER_INIT
    #   TRAINING_LOOP
    #     TRAINING_SINGLE_ITERATION
    #       DATA_LOADING
    #       MODEL_FORWARD
    #       ZERO_GRAD
    #       MODEL_BACKWARD
    #       OPTIMIZER_UPDATE
    #     CHECKPOINT_SAVE_SYNC
    #     TRAINING_METRICS_UPDATE
    #   VALIDATION_LOOP
    #     VALIDATION_SINGLE_ITERATION
    #   TESTING_LOOP
    #     TESTING_SINGLE_ITERATION
    mock_time.return_value = 45000.0
    mock_perf_counter.return_value = 1000.0

    training_recorder.on_app_start(start_time=TracingTimestamp.now())
    latest_ts: TracingTimestamp = advance_time(mock_time, mock_perf_counter, 10.0)

    training_recorder.on_dataloader_init_start(start_time=latest_ts)
    latest_ts = advance_time(mock_time, mock_perf_counter, 30.0)
    training_recorder.on_dataloader_init_end(stop_time=latest_ts)
    latest_ts = advance_time(mock_time, mock_perf_counter, 50.0)

    training_recorder.on_load_checkpoint_start(start_time=latest_ts)
    latest_ts = advance_time(mock_time, mock_perf_counter, 20.0)
    training_recorder.on_load_checkpoint_end(stop_time=latest_ts)

    training_recorder.on_model_init_start(start_time=latest_ts)
    latest_ts = advance_time(mock_time, mock_perf_counter, 20.0)
    training_recorder.on_model_init_end(stop_time=latest_ts)

    training_recorder.on_optimizer_init_start(start_time=latest_ts)
    latest_ts = advance_time(mock_time, mock_perf_counter, 40.0)
    training_recorder.on_optimizer_init_end(stop_time=latest_ts)

    training_recorder.on_training_loop_start(train_iterations_start=0, train_samples_start=0)
    for training_iteration in range(num_train_iterations):
        training_recorder.on_training_single_iteration_start(start_time=latest_ts)
        data_loading_span = training_recorder.start(StandardTrainingJobSpanName.DATA_LOADING)
        latest_ts = advance_time(mock_time, mock_perf_counter, 40.0)
        training_recorder.stop(data_loading_span)

        model_forward_span = training_recorder.start(StandardTrainingJobSpanName.MODEL_FORWARD)
        latest_ts = advance_time(mock_time, mock_perf_counter, 10.0)
        training_recorder.stop(model_forward_span)

        zero_grad_span = training_recorder.start(StandardTrainingJobSpanName.ZERO_GRAD)
        latest_ts = advance_time(mock_time, mock_perf_counter, 20.0)
        training_recorder.stop(zero_grad_span)

        model_backward_span = training_recorder.start(StandardTrainingJobSpanName.MODEL_BACKWARD)
        latest_ts = advance_time(mock_time, mock_perf_counter, 10.0)
        training_recorder.stop(model_backward_span)

        optimizer_update_span = training_recorder.start(StandardTrainingJobSpanName.OPTIMIZER_UPDATE)
        latest_ts = advance_time(mock_time, mock_perf_counter, 5.0)
        training_recorder.stop(optimizer_update_span)

        latest_ts = advance_time(mock_time, mock_perf_counter, 100.0)
        training_recorder.on_training_single_iteration_end(stop_time=latest_ts)

        if training_iteration > 0 and training_iteration % checkpoint_interval == 0:
            training_recorder.on_save_checkpoint_start(current_iteration=training_iteration, start_time=latest_ts)
            latest_ts = advance_time(mock_time, mock_perf_counter, 50.0)
            training_recorder.on_save_checkpoint_success(current_iteration=training_iteration, timestamp=latest_ts)
            training_recorder.on_save_checkpoint_end(stop_time=latest_ts)
        if training_iteration > 0 and training_iteration % validation_interval == 0:
            training_recorder.on_validation_start(start_time=latest_ts)
            training_recorder.on_validation_single_iteration_start(start_time=latest_ts)
            latest_ts = advance_time(mock_time, mock_perf_counter, 40.0)
            training_recorder.on_validation_single_iteration_end(stop_time=latest_ts)
            training_recorder.on_validation_end(stop_time=latest_ts)

    training_recorder.on_training_loop_end(stop_time=latest_ts)

    training_recorder.on_testing_start(start_time=latest_ts)
    for _ in range(5):
        testing_single_iteration_span = training_recorder.start(StandardTrainingJobSpanName.TESTING_SINGLE_ITERATION)
        training_recorder.stop(testing_single_iteration_span)

    latest_ts = advance_time(mock_time, mock_perf_counter, 40.0)
    training_recorder.on_testing_end(stop_time=latest_ts)
    training_recorder.on_app_end(stop_time=latest_ts)

    mock_exporter: Mock = training_recorder._exporters[0]  # type: ignore[assignment]

    assert mock_exporter.initialize.call_count == 1

    span_names = [c.args[0].name_str for c in mock_exporter.export_start.mock_calls]
    expected_exported_span_names_freq = {
        StandardSpanName.APPLICATION: 1,
        StandardTrainingJobSpanName.DATA_LOADER_INIT: 1,
        StandardTrainingJobSpanName.CHECKPOINT_LOAD: 1,
        StandardTrainingJobSpanName.MODEL_INIT: 1,
        StandardTrainingJobSpanName.OPTIMIZER_INIT: 1,
        StandardTrainingJobSpanName.TRAINING_LOOP: 1,
        StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC: 29,
        StandardTrainingJobSpanName.VALIDATION_LOOP: 17,
        StandardTrainingJobSpanName.TESTING_LOOP: 1,
    }
    for span_name, freq in expected_exported_span_names_freq.items():
        if span_names.count(span_name) != freq:
            print(f"Span {span_name} was exported {span_names.count(span_name)} times, expected {freq} times")
        assert span_names.count(span_name) == freq
    # Nothing outside of the expected span names was exported
    assert set(span_names) - set(expected_exported_span_names_freq.keys()) == set()

    expected_exported_event_names_freq = {
        StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION: 1,
        StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS: 29,
        StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE: 29,
        # reporting training metrics every 10 iterations (10, 20, ...80)
        StandardTrainingJobEventName.TRAINING_METRICS_UPDATE: 8,
        StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE: 17,
        StandardTrainingJobEventName.TESTING_METRICS_UPDATE: 1,
    }
    event_names = [c.args[0].name_str for c in mock_exporter.export_event.mock_calls]
    for event_name, freq in expected_exported_event_names_freq.items():
        assert event_names.count(event_name) == freq
    # Nothing outside of the expected event names was exported
    assert set(event_names) - set(expected_exported_event_names_freq.keys()) == set()
