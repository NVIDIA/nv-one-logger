# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the context module."""
# pyright: reportPrivateUsage=false

from typing import Dict
from unittest.mock import MagicMock, Mock

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.attributes import Attributes, AttributeValue
from nv_one_logger.core.event import StandardEventAttributeName
from nv_one_logger.core.span import StandardSpanAttributeName, StandardSpanName
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.training_telemetry.api.attributes import (
    CheckpointSaveSpanAttributes,
    SaveCheckpointSuccessEventAttributes,
    SyncCheckpointMetricsUpdateAttributes,
    ValidationMetricsUpdateAttributes,
)
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.context import (
    application,
    checkpoint_load,
    checkpoint_save,
    dataloader_init,
    model_init,
    optimizer_init,
    testing_loop,
    training_iteration,
    training_loop,
    validation_iteration,
    validation_loop,
)
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from .conftest import configure_provider_for_test
from .utils import (
    advance_time,
    assert_exporter_method_call_sequence,
    assert_only_start_stop_event,
    get_non_trivial_events,
    reset_singletong_providers_for_test,
    span_from_export_start,
    span_from_export_stop,
)

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


def test_application_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the application context manager creates and stops the appropriate spans."""
    with application() as span:
        advance_time(mock_time, mock_perf_counter, 700.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardSpanName.APPLICATION
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 700000})

    event = get_non_trivial_events(span)
    assert len(event) == 2
    assert event[0].name == StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION
    assert event[1].name == StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG
    assert mock_exporter.export_event.call_count == 2

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,  # ONE_LOGGER_INITIALIZATION
            Exporter.export_event,  # UPDATE_TRAINING_TELEMETRY_CONFIG
            Exporter.export_stop,
            Exporter.close,
        ],
    )


def test_training_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, one_logger_config: OneLoggerConfig) -> None:
    """Test that the training context manager creates and stops the appropriate spans."""
    assert one_logger_config.telemetry_config
    with training_loop(
        train_iterations_start=0, train_iterations_target_or_fn=1000, train_samples_target_or_fn=1000 * one_logger_config.telemetry_config.global_batch_size
    ) as span:
        advance_time(mock_time, mock_perf_counter, 700.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span.name == StandardTrainingJobSpanName.TRAINING_LOOP
    expected_attributes: Dict[str, AttributeValue] = {
        "completed_floating_point_operations_overall": 0,
        "train_iterations_start": 0,
        "train_iterations_target": 1000,
        "train_samples_start": 0,
        "train_samples_target": 32000,  # train_iterations_target * global_batch_size
        StandardSpanAttributeName.DURATION_MSEC: 700000,
    }
    assert span.attributes == Attributes(expected_attributes)
    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_training_iteration_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock, one_logger_config: OneLoggerConfig) -> None:
    """Test that the training iteration context manager creates and stops the appropriate spans."""
    # Start a training loop first
    with training_loop(train_iterations_start=0):
        with training_iteration():
            advance_time(mock_time, mock_perf_counter, 500.0)

    # Verify that only the training loop span was created (training iteration doesn't export spans)
    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,  # Training loop
            Exporter.export_stop,  # Training loop
        ],
    )


def test_validation_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the validation context manager creates and stops the appropriate spans."""
    with validation_loop() as span:
        advance_time(mock_time, mock_perf_counter, 20)
        for i in range(10):
            with validation_iteration():
                advance_time(mock_time, mock_perf_counter, 50.0 + (i % 2 * 10))

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.VALIDATION_LOOP
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: (20 + 550) * 1000})

    assert mock_exporter.export_event.call_count == 1
    events = get_non_trivial_events(span)
    assert len(events) == 1
    event = events[0]
    assert event.name == StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE
    assert event.attributes == ValidationMetricsUpdateAttributes.create(
        current_iteration=0,
        interval=0,
        avg_iteration_time_sec=55.0,
        min_iteration_time_sec=50.0,
        max_iteration_time_sec=60.0,
        total_iteration_time_sec=550.0,
    ).add(StandardEventAttributeName.TIMESTAMP_MSEC, (STARTING_TIME + 20 + 550) * 1000)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_testing_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the testing context manager creates and stops the appropriate spans."""
    with testing_loop():
        advance_time(mock_time, mock_perf_counter, 900.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.TESTING_LOOP
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 900000})

    assert mock_exporter.export_event.call_count == 1
    events = get_non_trivial_events(span)
    assert len(events) == 1
    event = events[0]
    assert event.name == StandardTrainingJobEventName.TESTING_METRICS_UPDATE
    assert event.attributes == Attributes({StandardEventAttributeName.TIMESTAMP_MSEC: (STARTING_TIME + 900) * 1000, "interval": 0, "current_iteration": 0})

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_checkpoint_load_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the checkpoint load context manager creates and stops the appropriate spans."""
    with checkpoint_load() as span:
        advance_time(mock_time, mock_perf_counter, 700.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_LOAD
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 700000})
    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_checkpoint_save_context_success(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the checkpoint save context manager creates and stops the appropriate spans when the checkpoint save is successful."""
    global_step = 100

    with checkpoint_save(global_step) as span:
        advance_time(mock_time, mock_perf_counter, 200.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
    expected_attributes = CheckpointSaveSpanAttributes.create(
        checkpoint_strategy=CheckPointStrategy.SYNC, current_iteration=global_step, save_checkpoint_attempt_count=1
    )
    expected_attributes.add(StandardSpanAttributeName.DURATION_MSEC, 200000)

    assert span.attributes == expected_attributes

    assert mock_exporter.export_event.call_count == 2
    events = get_non_trivial_events(span)
    assert len(events) == 2

    ckpt_success_event = events[0]
    assert ckpt_success_event.name == StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS
    expected_success_ev_attributes = SaveCheckpointSuccessEventAttributes.create(
        checkpoint_strategy=CheckPointStrategy.SYNC,
        current_iteration=global_step,
        first_successful_save_checkpoint_timestamp_sec=STARTING_TIME + 200,
        latest_successful_save_checkpoint_timestamp_sec=STARTING_TIME + 200,
        save_checkpoint_success_count=1,
        productive_train_iterations=0,
        productive_train_samples=0,
        productive_train_iterations_sec=0,
        productive_validation_iterations_sec=0,
        productive_train_tflops=None,
        training_start_timestamp_sec=0,
    )
    expected_success_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, int((STARTING_TIME + 200) * 1000))
    assert ckpt_success_event.attributes == expected_success_ev_attributes

    ckpt_sync_metrics_update_event = events[1]
    assert ckpt_sync_metrics_update_event.name == StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE
    expected_metrics_ev_attributes = SyncCheckpointMetricsUpdateAttributes.create(
        save_checkpoint_sync_time_total_sec=200.0,
        save_checkpoint_sync_time_min_sec=200.0,
        save_checkpoint_sync_time_max_sec=200.0,
    )
    expected_metrics_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, int((STARTING_TIME + 200) * 1000))
    assert ckpt_sync_metrics_update_event.attributes == expected_metrics_ev_attributes

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_event,
            Exporter.export_event,
            Exporter.export_stop,
        ],
    )


def test_checkpoint_save_context_failure(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the checkpoint save context manager creates and stops the appropriate spans when the checkpoint save fails."""
    global_step = 100

    with pytest.raises(Exception, match="Checkpoint save failed"):
        with checkpoint_save(global_step) as span:
            advance_time(mock_time, mock_perf_counter, 40.0)
            raise Exception("Checkpoint save failed")

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
    expected_attributes = CheckpointSaveSpanAttributes.create(
        checkpoint_strategy=CheckPointStrategy.SYNC, current_iteration=global_step, save_checkpoint_attempt_count=1
    )
    expected_attributes.add(StandardSpanAttributeName.DURATION_MSEC, 40000)
    assert span.attributes == expected_attributes

    assert mock_exporter.export_event.call_count == 1
    events = get_non_trivial_events(span)
    assert len(events) == 1
    event = events[0]
    assert event.name == StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE
    expected_ev_attributes = SyncCheckpointMetricsUpdateAttributes.create(
        save_checkpoint_sync_time_total_sec=40.0,
        save_checkpoint_sync_time_min_sec=40.0,
        save_checkpoint_sync_time_max_sec=40.0,
    )
    expected_ev_attributes.add(StandardEventAttributeName.TIMESTAMP_MSEC, int((STARTING_TIME + 40) * 1000))
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


def test_model_init_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the model initialization context manager creates and stops the appropriate spans."""
    with model_init() as span:
        advance_time(mock_time, mock_perf_counter, 300.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.MODEL_INIT
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 300000})
    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_dataloader_init_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the dataloader initialization context manager creates and stops the appropriate spans."""
    with dataloader_init() as span:
        advance_time(mock_time, mock_perf_counter, 400.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.DATA_LOADER_INIT
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 400000})
    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_optimizer_init_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the optimizer initialization context manager creates and stops the appropriate spans."""
    with optimizer_init() as span:
        advance_time(mock_time, mock_perf_counter, 250.0)

    assert mock_exporter.export_start.call_count == 1
    assert mock_exporter.export_stop.call_count == 1
    span = span_from_export_start(mock_exporter, expected_parent=None)
    assert span == span_from_export_stop(mock_exporter)
    assert span.name == StandardTrainingJobSpanName.OPTIMIZER_INIT
    assert span.attributes == Attributes({StandardSpanAttributeName.DURATION_MSEC: 250000})
    assert_only_start_stop_event(span, mock_exporter)

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
            Exporter.export_start,
            Exporter.export_stop,
        ],
    )


def test_validation_iteration_context(mock_exporter: MagicMock, mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that the validation iteration context manager creates and stops the appropriate spans."""
    with validation_iteration():
        advance_time(mock_time, mock_perf_counter, 150.0)

    # No data is exported for the validation iteration span.
    assert mock_exporter.export_start.call_count == 0
    assert mock_exporter.export_stop.call_count == 0

    assert_exporter_method_call_sequence(
        mock_exporter,
        [
            Exporter.initialize,
        ],
    )


def test_disabled_for_current_rank(one_logger_config: OneLoggerConfig, mock_exporter: MagicMock) -> None:
    """Test that the training telemetry is disabled for the current rank."""
    one_logger_config.enable_for_current_rank = False
    reset_singletong_providers_for_test()
    (TrainingTelemetryProvider.instance().with_base_config(one_logger_config).with_exporter(mock_exporter).configure_provider())

    # Try a few context managers to make sure the provider is disabled.
    with application() as app_span:
        assert app_span is None
        with model_init() as model_init_span:  # type: ignore[unreachable]
            assert model_init_span is None
        with dataloader_init() as dataloader_init_span:
            assert dataloader_init_span is None
        with optimizer_init() as optimizer_init_span:
            assert optimizer_init_span is None
        with checkpoint_load() as checkpoint_load_span:
            assert checkpoint_load_span is None
        with checkpoint_save(100) as checkpoint_save_span:
            assert checkpoint_save_span is None
        with training_loop(
            train_iterations_start=0,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=32000,
        ) as training_loop_span:
            assert training_loop_span is None
            with training_iteration() as training_iteration_span:
                assert training_iteration_span is None
        with validation_loop() as validation_loop_span:
            assert validation_loop_span is None
            with validation_iteration() as validation_iteration_span:
                assert validation_iteration_span is None
        with testing_loop() as testing_loop_span:
            assert testing_loop_span is None
    mock_exporter.assert_not_called()  # type: ignore[unreachable]

    # Undo the force disable logging so that other tests don't fail.
    OneLoggerProvider.instance()._logging_force_disabled = False
