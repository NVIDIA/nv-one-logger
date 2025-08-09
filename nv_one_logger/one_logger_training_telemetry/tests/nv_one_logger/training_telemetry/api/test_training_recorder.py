# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TrainingRecorder class."""
# pyright: reportPrivateUsage=false

from typing import Set
from unittest.mock import Mock

import pytest
from nv_one_logger.core.event import StandardEventAttributeName, StandardEventName
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.metric_summarizer import MetricSummarizer
from nv_one_logger.core.internal.multi_window_timer import MultiWindowTimer
from nv_one_logger.core.span import Span, SpanName, StandardSpanName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter

from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder, _ProductivityState, _TrainingState
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from .conftest import configure_provider_for_test
from .utils import advance_time, span_from_export_start

STARTING_PERF_COUNTER = 5000.0
STARTING_TIME = 120000.0


@pytest.fixture(autouse=True)
def configure_provider(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
    """Fixture that configures the TrainingTelemetryProvider."""
    configure_provider_for_test(config, mock_exporter)


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
    return TrainingTelemetryProvider.instance().recorder


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
    assert state.train_samples_processed_current_job == 0
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
    train_samples_start = 3000
    training_recorder.on_training_loop_start(train_iterations_start=train_iterations_start, train_samples_start=train_samples_start)
    assert config.training_loop_config is not None
    world_size = config.training_loop_config.world_size
    global_batch_size = config.training_loop_config.global_batch_size
    flops_per_sample = config.training_loop_config.flops_per_sample

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
        train_samples_start=train_samples_start,
        train_samples_processed_current_job=0,
        total_flops_current_job=0,
        train_tokens_current_job=None,
        completed_floating_point_operations_overall=train_iterations_start * global_batch_size * flops_per_sample,
        training_loop_start_time=TracingTimestamp.now(),
        first_logged_train_iterations_finish_time=None,
        last_logged_train_iterations_finish_time=None,
        first_save_checkpoint_success_time=None,
        latest_save_checkpoint_success_time=None,
        validation_interval_start=5,
        testing_interval_start=5,
        tflops_per_gpu=MetricSummarizer[float](),
        successful_save_checkpoint_count_current_job=0,
        productivity_state={},
        max_reported_productive_train_iterations=-1,
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
    expected_state.train_samples_processed_current_job = 32  # batch size
    expected_state.first_logged_train_iterations_finish_time = latest_ts
    expected_state.last_logged_train_iterations_finish_time = latest_ts
    expected_state.tflops_per_gpu.add_value(3200 / (10 * 10**12 * world_size))
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
    expected_state.train_samples_processed_current_job = 32 * 2
    expected_state.last_logged_train_iterations_finish_time = latest_ts
    expected_state.tflops_per_gpu.add_value(3200 * 2 / ((10 + 20) * 10**12 * world_size))
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
    expected_state.train_samples_processed_current_job = 32 * 3
    expected_state.last_logged_train_iterations_finish_time = latest_ts
    expected_state.tflops_per_gpu.add_value(3200 * 3 / ((10 + 20 + 20) * 10**12 * world_size))
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
    expected_state.productivity_state = {
        2: _ProductivityState(
            productive_train_iterations=train_iterations_start + 3,
            productive_train_samples=train_samples_start + 32 * 3,
            productive_train_iterations_sec=10 + 20 + 20,
            productive_validation_iterations_sec=20 + 50,
            productive_train_tflops=(train_iterations_start + 3) * global_batch_size * flops_per_sample / 10**12,
        )
    }
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
    assert config.training_loop_config is not None
    assert config.training_loop_config.log_every_n_train_iterations == 10
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
        StandardTrainingJobEventName.TRAINING_METRICS_UPDATE: 9,
        StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE: 17,
        StandardTrainingJobEventName.TESTING_METRICS_UPDATE: 1,
    }
    event_names = [c.args[0].name_str for c in mock_exporter.export_event.mock_calls]
    for event_name, freq in expected_exported_event_names_freq.items():
        assert event_names.count(event_name) == freq
    # Nothing outside of the expected event names was exported
    assert set(event_names) - set(expected_exported_event_names_freq.keys()) == set()
    assert set(event_names) - set(expected_exported_event_names_freq.keys()) == set()


def test_timer_auto_stop_error_recording(training_recorder: TrainingRecorder, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that timer auto-stop functionality records an error event on the span.

    This test verifies that when a span is stopped while its associated timer is still active,
    the timer is automatically stopped and an error event is recorded on the span to indicate
    that the timer was forced to stop.

    The corner case could be:
    - if an application hit some application-side failure and crashes
    - e.g., load_checkpoint process breaks and raise exception before `on_load_checkpoint_end` gets called because of checkpoint file not found.
    - The LOAD_CHECKPOINT span will be stopped by nv-one-logger for cleanning up and the timer will be forced to stop.
    - We need to make sure that the error event is recorded on the span to indicate that the timer was forced to stop.

    Args:
        training_recorder: A configured TrainingRecorder instance.
        mock_perf_counter: Mock for perf_counter.
        mock_time: Mock for time.
    """
    # Start a checkpoint load operation
    start_time = TracingTimestamp.now()
    training_recorder.on_app_start(start_time=start_time)
    span = training_recorder.on_load_checkpoint_start(start_time)

    # Verify timer is active
    timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_LOAD]
    assert timer.is_active, "Timer should be active after starting checkpoint load"

    # Advance time
    latest_ts = advance_time(mock_time, mock_perf_counter, 10.0)

    # Stop the span without calling on_load_checkpoint_end (simulating an error)
    # This should automatically stop the timer and record an error event
    training_recorder.on_app_end(stop_time=latest_ts)

    # Verify timer is no longer active
    assert not timer.is_active, "Timer should be stopped after stopping span"

    # Verify span is stopped
    assert not span.active, "Span should be stopped"

    # Verify span has stop event
    assert span.stop_event is not None, "Span should have a stop event"

    # Verify that an error event was recorded on the span
    error_events = [event for event in span.events if event.name == "error"]
    assert len(error_events) == 1, "Span should have exactly one error event"

    error_event = error_events[0]
    assert "Timer for span" in error_event.error_message, "Error message should mention timer"
    assert "automatically stopped" in error_event.error_message, "Error message should mention automatic stop"
    assert "checkpoint_load" in error_event.error_message, "Error message should mention the span name"

    # Verify the error event was exported
    mock_exporter: Mock = training_recorder._exporters[0]  # type: ignore[assignment]
    error_export_calls = [call for call in mock_exporter.export_error.mock_calls if call.args[1] == span]
    assert len(error_export_calls) == 1, "Error event should be exported exactly once"

    exported_error_event = error_export_calls[0].args[0]
    assert exported_error_event == error_event, "Exported error event should match the recorded error event"


@pytest.mark.parametrize(
    "span_name,start_method,expected_error_message,time_advance,config_setup",
    [
        (
            StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION,
            lambda recorder, start_time: recorder.on_training_single_iteration_start(start_time),
            "training_single_iteration",
            5.0,
            None,
        ),
        (
            StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION,
            lambda recorder, start_time: recorder.on_validation_single_iteration_start(start_time),
            "validation_single_iteration",
            3.0,
            None,
        ),
        (
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC,
            lambda recorder, start_time: recorder.on_save_checkpoint_start(current_iteration=1, start_time=start_time),
            "checkpoint_save_sync",
            15.0,
            None,
        ),
        (
            StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC,
            lambda recorder, start_time: recorder.on_save_checkpoint_start(current_iteration=1, start_time=start_time),
            "checkpoint_save_async",
            8.0,
            "async",
        ),
    ],
)
def test_timer_auto_stop_for_spans(
    span_name: StandardTrainingJobSpanName,
    start_method,
    expected_error_message: str,
    time_advance: float,
    config_setup: str,
    training_recorder: TrainingRecorder,
    mock_perf_counter: Mock,
    mock_time: Mock,
) -> None:
    """Test timer auto-stop functionality for various spans.

    This test verifies that when a span is stopped while its timer is still active,
    the timer is automatically stopped and an error event is recorded.

    Args:
        span_name: The span name to test.
        start_method: Function to start the span.
        expected_error_message: Expected error message fragment.
        time_advance: Time to advance in the test.
        config_setup: Configuration setup type (None for default, "async" for async checkpoint).
        training_recorder: A configured TrainingRecorder instance.
        mock_perf_counter: Mock for perf_counter.
        mock_time: Mock for time.
    """
    # Setup async config if needed
    if config_setup == "async":
        from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        async_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            save_checkpoint_strategy=CheckPointStrategy.ASYNC,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=10,
                train_iterations_target_or_fn=100,
                train_samples_target_or_fn=3200,
            ),
        )

        # Reconfigure the provider with async config
        from .conftest import configure_provider_for_test

        mock_exporter = training_recorder._exporters[0]
        configure_provider_for_test(async_config, mock_exporter)
        training_recorder = TrainingTelemetryProvider.instance().recorder

    # Start the span
    start_time = TracingTimestamp.now()
    training_recorder.on_app_start(start_time=start_time)
    span = start_method(training_recorder, start_time)

    # Verify timer is active
    timer = training_recorder._training_state.multi_iteration_timers[span_name]
    assert timer.is_active, f"Timer should be active after starting {span_name}"

    # Advance time
    latest_ts = advance_time(mock_time, mock_perf_counter, time_advance)

    # Stop the span without calling the corresponding end method (simulating an error)
    training_recorder.on_app_end(stop_time=latest_ts)

    # Verify timer is no longer active
    assert not timer.is_active, f"Timer should be stopped after stopping {span_name}"

    # Verify that an error event was recorded on the span
    error_events = [event for event in span.events if event.name == "error"]
    assert len(error_events) == 1, f"Span should have exactly one error event for {span_name}"

    error_event = error_events[0]
    assert expected_error_message in error_event.error_message, f"Error message should mention {expected_error_message}"


def test_timer_auto_stop_no_error_when_timer_already_stopped(training_recorder: TrainingRecorder, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that no error event is recorded when timer is already stopped.

    This test verifies that when a span is stopped and its associated timer is already stopped,
    no error event is recorded since the timer was properly stopped.

    Args:
        training_recorder: A configured TrainingRecorder instance.
        mock_perf_counter: Mock for perf_counter.
        mock_time: Mock for time.
    """
    # Start a checkpoint load operation
    start_time = TracingTimestamp.now()
    training_recorder.on_app_start(start_time=start_time)
    span = training_recorder.on_load_checkpoint_start(start_time)

    # Verify timer is active
    timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_LOAD]
    assert timer.is_active, "Timer should be active after starting checkpoint load"

    # Advance time and properly stop the timer first
    latest_ts = advance_time(mock_time, mock_perf_counter, 10.0)
    training_recorder.on_load_checkpoint_end(stop_time=latest_ts)
    training_recorder.on_app_end(stop_time=latest_ts)

    # Verify timer is no longer active
    assert not timer.is_active, "Timer should be stopped after calling on_load_checkpoint_end"

    # The span is already stopped by on_load_checkpoint_end, so we can't stop it again
    # Instead, let's verify that no error events were recorded during the proper stop
    error_events = [event for event in span.events if event.name == "error"]
    assert len(error_events) == 0, "No error events should be recorded when timer is properly stopped"


def test_timer_auto_stop_for_span_without_timer(training_recorder: TrainingRecorder, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that stopping a span without an associated timer doesn't cause issues.

    This test verifies that stopping a span that doesn't have an associated timer
    doesn't trigger any timer-related logic and doesn't cause errors.

    Args:
        training_recorder: A configured TrainingRecorder instance.
        mock_perf_counter: Mock for perf_counter.
        mock_time: Mock for time.
    """
    # Start an application span (which doesn't have an associated timer)
    start_time = TracingTimestamp.now()
    span = training_recorder.on_app_start(start_time)

    # Verify the span name is not in the multi-iteration timers
    assert span.name not in training_recorder._training_state.multi_iteration_timers, "Application span should not have an associated timer"

    # Advance time
    latest_ts = advance_time(mock_time, mock_perf_counter, 5.0)

    # Stop the span
    training_recorder.stop(span, stop_time=latest_ts)

    # Verify span is stopped
    assert not span.active, "Span should be stopped"

    # Verify no error events were recorded (since there's no timer to auto-stop)
    error_events = [event for event in span.events if event.name == "error"]
    assert len(error_events) == 0, "No error events should be recorded for spans without timers"


def test_timer_auto_stop_error_message_content(training_recorder: TrainingRecorder, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the error message content is correct and informative.

    This test verifies that the error message recorded when a timer is auto-stopped
    contains the expected information and is helpful for debugging.

    Args:
        training_recorder: A configured TrainingRecorder instance.
        mock_perf_counter: Mock for perf_counter.
        mock_time: Mock for time.
    """
    # Start a training single iteration
    start_time = TracingTimestamp.now()
    training_recorder.on_app_start(start_time=start_time)
    span = training_recorder.on_training_single_iteration_start(start_time)

    # Advance time
    latest_ts = advance_time(mock_time, mock_perf_counter, 7.0)

    # Stop the span without calling the proper end method
    training_recorder.on_app_end(stop_time=latest_ts)

    # Get the error event
    error_events = [event for event in span.events if event.name == "error"]
    assert len(error_events) == 1, "Span should have exactly one error event"

    error_event = error_events[0]
    error_message = error_event.error_message

    # Verify the error message contains all expected components
    assert "Timer for span" in error_message, "Error message should mention timer"
    assert "training_single_iteration" in error_message, "Error message should mention the span name"
    assert "automatically stopped" in error_message, "Error message should mention automatic stop"
    assert "span is being stopped" in error_message, "Error message should mention span stopping"
    assert "on_xxx_end method was not called correctly" in error_message, "Error message should mention missing end method"


def test_app_crash_scenario_with_open_spans(training_recorder: TrainingRecorder, mock_exporter: Mock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test what happens when the app crashes after certain events are called.

    This test simulates the scenario where:
    1. on_app_start is called
    2. on_training_loop_start is called
    3. on_training_single_iteration_start is called
    4. The app crashes (simulated by not calling the corresponding *_end methods)

    Expected behavior:
    - All spans remain open (not stopped)
    - No cleanup occurs
    - Active spans are not exported as stopped
    - The recorder is not closed
    """
    # Step 1: Start the application
    app_span = training_recorder.on_app_start(start_time=TracingTimestamp.now())
    assert len(training_recorder._spans) == 1
    assert training_recorder._get_active_span(StandardSpanName.APPLICATION) == app_span
    assert len(app_span.events) == 2  # Start event + initialization event
    assert app_span.events[0].name == StandardEventName.SPAN_START

    # Verify the app span was exported as started
    mock_exporter.export_start.assert_called_once()
    exported_app_span = span_from_export_start(mock_exporter, expected_parent=None)
    assert exported_app_span == app_span

    # Step 2: Start the training loop
    advance_time(mock_time, mock_perf_counter, 1.0)  # Advance time by 1 second
    training_loop_span = training_recorder.on_training_loop_start(
        train_iterations_start=0, train_samples_start=0, train_iterations_target=100, train_samples_target=3200, start_time=TracingTimestamp.now()
    )
    assert len(training_recorder._spans) == 2
    assert training_recorder._get_active_span(StandardTrainingJobSpanName.TRAINING_LOOP) == training_loop_span
    assert len(training_loop_span.events) == 1  # Only start event
    assert training_loop_span.events[0].name == StandardEventName.SPAN_START

    # Verify the training loop span was exported as started
    assert mock_exporter.export_start.call_count == 2
    exported_training_loop_span = span_from_export_start(mock_exporter, expected_parent=app_span)
    assert exported_training_loop_span == training_loop_span

    # Step 3: Start a training iteration
    advance_time(mock_time, mock_perf_counter, 0.5)  # Advance time by 0.5 seconds

    # Check if the timer is already active (it shouldn't be)
    training_iteration_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
    assert not training_iteration_timer.is_active, "Training iteration timer should not be active before starting"

    training_iteration_span = training_recorder.on_training_single_iteration_start(start_time=TracingTimestamp.now())
    assert len(training_recorder._spans) == 3
    assert training_recorder._get_active_span(StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION) == training_iteration_span
    assert len(training_iteration_span.events) == 1  # Only start event
    assert training_iteration_span.events[0].name == StandardEventName.SPAN_START

    # Note: Training iteration spans are not exported by default due to DEFAULT_SPANS_EXPORT_BLACKLIST
    # This is because they occur frequently and would result in a lot of data
    assert mock_exporter.export_start.call_count == 2  # Only app and training_loop spans exported

    # Step 4: Simulate app crash - NO cleanup calls are made
    # This is the key part: we don't call any of the *_end methods

    # Verify that no spans were stopped
    assert mock_exporter.export_stop.call_count == 0

    # Verify all spans are still active
    assert len(training_recorder._spans) == 3
    assert training_recorder._get_active_span(StandardSpanName.APPLICATION) == app_span
    assert training_recorder._get_active_span(StandardTrainingJobSpanName.TRAINING_LOOP) == training_loop_span
    assert training_recorder._get_active_span(StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION) == training_iteration_span

    # Verify that the recorder is not closed
    assert not training_recorder._closed

    # Verify that no telemetry data errors were reported
    mock_exporter.export_telemetry_data_error.assert_not_called()

    # Step 5: Verify the state of each span
    # All spans should be active (not stopped)
    # Note: app_span has 2 events (start + initialization), others have 1 (start only)
    assert len(app_span.events) == 2
    assert app_span.events[0].name == StandardEventName.SPAN_START
    assert app_span.active  # App span is still active
    assert app_span.stop_event is None  # App span has no stop event

    for span in [training_loop_span, training_iteration_span]:
        assert len(span.events) == 1
        assert span.events[0].name == StandardEventName.SPAN_START
        assert span.active  # Spans are still active
        assert span.stop_event is None  # Spans have no stop events


def test_app_crash_scenario_with_manual_cleanup(training_recorder: TrainingRecorder, mock_exporter: Mock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test what happens when manual cleanup is performed after a crash scenario.

    This test demonstrates the proper cleanup sequence that should be called
    to handle incomplete spans after a crash.
    """
    # Step 1: Simulate the crash scenario (same as above)
    app_span = training_recorder.on_app_start(start_time=TracingTimestamp.now())
    advance_time(mock_time, mock_perf_counter, 1.0)
    training_loop_span = training_recorder.on_training_loop_start(
        train_iterations_start=0, train_samples_start=0, train_iterations_target=100, train_samples_target=3200, start_time=TracingTimestamp.now()
    )
    advance_time(mock_time, mock_perf_counter, 0.5)
    training_iteration_span = training_recorder.on_training_single_iteration_start(start_time=TracingTimestamp.now())

    # Verify we have 3 active spans
    assert len(training_recorder._spans) == 3
    assert mock_exporter.export_stop.call_count == 0

    # Step 2: Perform manual cleanup (what should happen in a proper shutdown)
    advance_time(mock_time, mock_perf_counter, 0.1)

    # Stop spans in reverse order (as done in DefaultRecorder.close())
    # Note: training_iteration_span is not exported due to blacklist, so stopping it won't call export_stop
    training_recorder.stop(training_iteration_span, stop_time=TracingTimestamp.now())
    assert mock_exporter.export_stop.call_count == 0  # Not exported due to blacklist

    training_recorder.stop(training_loop_span, stop_time=TracingTimestamp.now())
    assert mock_exporter.export_stop.call_count == 1

    training_recorder.stop(app_span, stop_time=TracingTimestamp.now())
    assert mock_exporter.export_stop.call_count == 2

    # Step 3: Close the recorder
    training_recorder.close()
    assert training_recorder._closed

    # Verify all exporters were closed
    mock_exporter.close.assert_called_once()

    # Verify all spans were cleared
    assert len(training_recorder._spans) == 0


def test_app_crash_scenario_with_context_managers(training_recorder: TrainingRecorder, mock_exporter: Mock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test what happens when context managers are used and an exception occurs.

    This test simulates using context managers and what happens when an exception
    is raised during execution.
    """
    from nv_one_logger.training_telemetry.api.context import application, training_iteration, training_loop

    # Step 1: Use context managers and simulate a crash with an exception
    try:
        with application():
            with training_loop(train_iterations_start=0, train_iterations_target_or_fn=100):
                with training_iteration():
                    # Advance time to ensure the timer has a positive duration
                    advance_time(mock_time, mock_perf_counter, 0.1)
                    # Simulate a crash by raising an exception
                    raise RuntimeError("Simulated crash during training iteration")
    except RuntimeError:
        # Exception was caught, but let's check the state
        pass

    # Step 2: Verify that context managers properly cleaned up
    # The finally blocks in context managers should have called the *_end methods
    assert mock_exporter.export_start.call_count == 2  # app, training_loop (training_iteration not exported due to blacklist)
    assert mock_exporter.export_stop.call_count == 2  # app, training_loop (training_iteration not exported)

    # Verify no active spans remain
    assert len(training_recorder._spans) == 0

    # Verify the recorder is closed (context managers do close the recorder via on_app_end)
    assert training_recorder._closed


def test_app_crash_scenario_with_telemetry_data_error_reporting(
    training_recorder: TrainingRecorder, mock_exporter: Mock, mock_perf_counter: Mock, mock_time: Mock
) -> None:
    """Test telemetry data error reporting in crash scenarios.

    This test demonstrates how to manually report incomplete telemetry data
    when a crash is detected.
    """
    # Step 1: Simulate the crash scenario
    training_recorder.on_app_start(start_time=TracingTimestamp.now())
    advance_time(mock_time, mock_perf_counter, 1.0)
    training_recorder.on_training_loop_start(
        train_iterations_start=0, train_samples_start=0, train_iterations_target=100, train_samples_target=3200, start_time=TracingTimestamp.now()
    )
    advance_time(mock_time, mock_perf_counter, 0.5)
    training_recorder.on_training_single_iteration_start(start_time=TracingTimestamp.now())

    # Verify we have active spans
    assert len(training_recorder._spans) == 3

    # Step 2: Manually report telemetry data error (what should be done in crash handling)
    from nv_one_logger.core.attributes import Attribute, Attributes

    attributes = Attributes()
    attributes.add_attribute(Attribute("crash_reason", "simulated_crash"))
    attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))

    training_recorder.telemetry_data_error("Application crashed with incomplete telemetry data", attributes=attributes)

    # Verify telemetry data error was reported
    mock_exporter.export_telemetry_data_error.assert_called_once()

    # Step 3: Verify the error event details
    error_call = mock_exporter.export_telemetry_data_error.call_args
    error_event = error_call.args[0]
    assert error_event.name == StandardEventName.TELEMETRY_DATA_ERROR
    assert "Application crashed with incomplete telemetry data" in str(error_event.attributes[StandardEventAttributeName.ERROR_MESSAGE])
    assert error_event.attributes["crash_reason"].value == "simulated_crash"
    assert error_event.attributes["active_spans_count"].value == 3
