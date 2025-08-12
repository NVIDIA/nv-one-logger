# SPDX-License-Identifier: Apache-2.0
"""Tests for training_recorder.py."""
from typing import Callable, Optional, Union

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.internal.safe_execution import safely_execute
from nv_one_logger.core.internal.utils import evaluate_value
from nv_one_logger.core.span import Span
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider


def _one_logger_config() -> OneLoggerConfig:
    return TrainingTelemetryProvider.instance().config


def _recorder() -> TrainingRecorder:
    return TrainingTelemetryProvider.instance().recorder


@safely_execute
def on_app_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the application is started.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting the application as milliseconds since epoch.
        If not provided, the current timestamp will be used as the start time of the application.

    Returns:
        The span corresponding to the entire application (StandardSpanName.APPLICATION).
    """
    start_time_ts: TracingTimestamp = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_app_start(start_time_ts)


@safely_execute
def on_app_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the application is about to end.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending the application as milliseconds since epoch.
        If not provided, the current timestamp will be used as the end time of the application.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_app_end(stop_time=stop_time)


@safely_execute
def on_model_init_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the model initialization starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting model initialization as milliseconds since epoch.
        If not provided, the current timestamp will be used as the start time of model initialization.

    Returns:
        The span corresponding to the model initialization (StandardTrainingJobSpanName.MODEL_INIT).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_model_init_start(start_time)


@safely_execute
def on_model_init_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the model initialization ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of the end of model initialization as milliseconds since epoch.
        If not provided, the current timestamp will be used as the end time of model initialization.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_model_init_end(stop_time)


@safely_execute
def on_dataloader_init_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the dataloader initialization starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting dataloader initialization as milliseconds since epoch.
        If not provided, the current timestamp will be used as the start time of dataloader initialization.

    Returns:
        The span corresponding to the dataloader initialization (StandardTrainingJobSpanName.DATA_LOADER_INIT).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_dataloader_init_start(start_time)


@safely_execute
def on_dataloader_init_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the dataloader initialization ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of the end of dataloader initialization as milliseconds since epoch.
        If not provided, the current timestamp will be used as the end time of dataloader initialization.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_dataloader_init_end(stop_time)


@safely_execute
def on_load_checkpoint_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the checkpoint loading starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting checkpoint loading as milliseconds since epoch.
        If not provided, the current timestamp will be used as the start time of checkpoint loading.

    Returns:
        The span corresponding to the checkpoint loading (StandardTrainingJobSpanName.CHECKPOINT_LOAD).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_load_checkpoint_start(start_time)


@safely_execute
def on_load_checkpoint_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the checkpoint loading ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of the end of checkpoint loading as milliseconds since epoch.
        If not provided, the current timestamp will be used as the end time of checkpoint loading.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_load_checkpoint_end(stop_time)


@safely_execute
def on_optimizer_init_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the optimizer initialization starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting optimizer initialization as milliseconds since epoch.
        If not provided, the current timestamp will be used as the start time of optimizer initialization.

    Returns:
        The span corresponding to the optimizer initialization (StandardTrainingJobSpanName.OPTIMIZER_INIT).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_optimizer_init_start(start_time)


@safely_execute
def on_optimizer_init_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the optimizer initialization ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of the end of optimizer initialization as milliseconds since epoch.
        If not provided, the current timestamp will be used as the end time of optimizer initialization.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_optimizer_init_end(stop_time)


@safely_execute
def on_train_start(
    train_iterations_start: int,
    train_samples_start: Optional[int] = None,
    train_iterations_target_or_fn: Optional[Union[int, Callable[[], int]]] = None,
    train_samples_target_or_fn: Optional[Union[int, Callable[[], int]]] = None,
    start_time_msec: Optional[float] = None,
) -> Span:
    """Call when the training loop starts.

    Args:
        train_iterations_start (int): The starting iteration number (could be non-zero if the job loads a checkpoint and starts from there).
        train_samples_start (Optional[int]): The starting sample number (could be non-zero if the job loads a checkpoint and starts from there).
            If not provided, the starting sample number will be calculated as `train_iterations_start * global_batch_size`.
        train_iterations_target (Optional[int]): Target number of training iterations or callable to generate it.
            If not provided, the target will be fetched from the config.
        train_samples_target (Optional[int]): Target number of training samples or function to generate the number.
            If not provided, the target will be fetched from the config.
        start_time_msec (Optional[float], optional): The timestamp of starting the training loop as milliseconds since epoch.
            If not provided, the current timestamp will be used as the start time of the training loop.

        NOTE: If the throughput logging is enabled, the target number of training iterations and samples must be provided
        either in the config or via this callback.

    Returns:
        The span corresponding to the training loop (StandardTrainingJobSpanName.TRAINING_LOOP).
    """
    conf = _one_logger_config()
    training_conf = conf.telemetry_config
    assert_that(training_conf is not None, "Training telemetry config must be set before starting the training loop.")
    assert_that(train_iterations_start is not None, "train_iterations_start is required.")  # type: ignore[reportUnnecessaryComparison]

    # Some of the input parameters need a bit of processing (e.g., setting a default for missing values).
    # Note: global_batch_size is now available from the UPDATE_TRAINING_TELEMETRY_CONFIG event posted to the application span
    train_samples_start = train_samples_start if train_samples_start is not None else train_iterations_start * training_conf.global_batch_size
    train_iterations_target: Optional[int] = None
    train_samples_target: Optional[int] = None
    train_tokens_target: Optional[int] = None
    if training_conf.is_log_throughput_enabled:
        # train_iterations_target and train_samples_target must be set either in the config or passed via this callback.
        train_iterations_target = (
            evaluate_value(train_iterations_target_or_fn) if train_iterations_target_or_fn is not None else training_conf.train_iterations_target
        )
        assert_that(train_iterations_target and train_iterations_target > 0, "train_iterations_target is required and must be a positive integer.")

        train_samples_target = evaluate_value(train_samples_target_or_fn) if train_samples_target_or_fn is not None else training_conf.train_samples_target
        assert_that(train_samples_target is not None and train_samples_target > 0, "train_samples_target is required and must be a positive integer.")

        if training_conf.seq_length and train_samples_target:
            train_tokens_target = training_conf.seq_length * train_samples_target

    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_training_loop_start(
        train_iterations_start=train_iterations_start,
        train_samples_start=train_samples_start,
        train_iterations_target=train_iterations_target,
        train_samples_target=train_samples_target,
        train_tokens_target=train_tokens_target,
        start_time=start_time,
    )


@safely_execute
def on_train_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when training loop ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending the training loop as milliseconds since epoch.
            If not provided, the current timestamp will be used as the end time of the training loop.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_training_loop_end(stop_time)


@safely_execute
def on_training_single_iteration_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when a training iteration starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting the training iteration as milliseconds since epoch.
            If not provided, the current timestamp will be used as the start time of the training iteration.

    Returns:
        The span corresponding to the training batch (StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_training_single_iteration_start(start_time)


@safely_execute
def on_training_single_iteration_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when a training iteration ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending the training iteration as milliseconds since epoch.
            If not provided, the current timestamp will be used as the end time of the training iteration.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_training_single_iteration_end(stop_time)


@safely_execute
def on_validation_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the validation loop starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting the validation loop as milliseconds since epoch.
            If not provided, the current timestamp will be used as the start time of the validation loop.

    Returns:
        The span corresponding to the validation loop (StandardTrainingJobSpanName.VALIDATION_LOOP).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_validation_start(start_time)


@safely_execute
def on_validation_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the validation loop ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending the validation loop as milliseconds since epoch.
            If not provided, the current timestamp will be used as the end time of the validation loop.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_validation_end(stop_time)


@safely_execute
def on_validation_single_iteration_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when a validation batch starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting the validation batch as milliseconds since epoch.
            If not provided, the current timestamp will be used as the start time of the validation batch.

    Returns:
        The span corresponding to the validation batch (StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_validation_single_iteration_start(start_time)


@safely_execute
def on_validation_single_iteration_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when a validation batch ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending the validation batch as milliseconds since epoch.
            If not provided, the current timestamp will be used as the end time of the validation batch.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_validation_single_iteration_end(stop_time)


@safely_execute
def on_testing_start(start_time_msec: Optional[float] = None) -> Span:
    """Call when the testing loop starts.

    Args:
        start_time_msec (Optional[float], optional): The timestamp of starting the testing loop as milliseconds since epoch.
            If not provided, the current timestamp will be used as the start time of the testing loop.

    Returns:
        The span corresponding to the testing loop (StandardTrainingJobSpanName.TESTING_LOOP).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_testing_start(start_time)


@safely_execute
def on_testing_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the testing loop ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending the testing loop as milliseconds since epoch.
            If not provided, the current timestamp will be used as the end time of the testing loop.
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_testing_end(stop_time)


@safely_execute
def on_save_checkpoint_start(global_step: int, start_time_msec: Optional[float] = None) -> Span:
    """Call when the checkpoint saving starts.

    Args:
        global_step (int): The global step (number of completed training iterations) at the time the checkpoint is saved.
        start_time_msec (Optional[float], optional): The timestamp of starting checkpoint saving as milliseconds since epoch.
            If not provided, the current timestamp will be used as the start time of checkpoint saving.

    Returns:
        The span corresponding to the checkpoint saving (CHECKPOINT_SAVE_SYNC or CHECKPOINT_SAVE_ASYNC).
    """
    start_time = TracingTimestamp.for_timestamp(timestamp_sec=start_time_msec / 1000.0) if start_time_msec else TracingTimestamp.now()
    return _recorder().on_save_checkpoint_start(current_iteration=global_step, start_time=start_time)


@safely_execute
def on_save_checkpoint_success(global_step: int, time_msec: Optional[float] = None) -> None:
    """Call when the checkpoint saving succeeds. Works both for sync and async checkpoint saving.

    This callback creates an event of type StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS and adds it to
    either CHECKPOINT_SAVE_SYNC span (for CheckPointStrategy.SYNC) or APPLICATION span (for CheckPointStrategy.ASYNC).
    See the comments on StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS for more details.

    Args:
        global_step (int): The global step (number of completed training iterations) at the time the checkpoint is saved.
        time_msec (Optional[float], optional): The timestamp of successful checkpoint saving as milliseconds since epoch.
            If not provided, the current timestamp will be used as the success time of checkpoint saving.
    """
    timestamp = TracingTimestamp.for_timestamp(timestamp_sec=time_msec / 1000.0) if time_msec else TracingTimestamp.now()
    _recorder().on_save_checkpoint_success(current_iteration=global_step, timestamp=timestamp)


@safely_execute
def on_save_checkpoint_end(finish_time_msec: Optional[float] = None) -> None:
    """Call when the checkpoint saving ends.

    Args:
        finish_time_msec (Optional[float], optional): The timestamp of ending checkpoint saving as milliseconds since epoch.
            If not provided, the current timestamp will be used as the end time of checkpoint saving.

    Returns:
        The span corresponding to the checkpoint saving (CHECKPOINT_SAVE_SYNC or CHECKPOINT_SAVE_ASYNC).
    """
    stop_time = TracingTimestamp.for_timestamp(timestamp_sec=finish_time_msec / 1000.0) if finish_time_msec else TracingTimestamp.now()
    _recorder().on_save_checkpoint_end(stop_time)
