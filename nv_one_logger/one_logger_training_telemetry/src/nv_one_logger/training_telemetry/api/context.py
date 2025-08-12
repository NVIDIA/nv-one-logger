# SPDX-License-Identifier: Apache-2.0
"""Context managers for recording events for the training loop.

Using these context managers is an alternative to calling the callbacks directly.
See README.md for more details.
"""

from contextlib import contextmanager
from typing import Callable, Generator, Optional, Union

from nv_one_logger.core.span import Span
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
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider


@contextmanager
def application() -> Generator[Span, None, None]:
    """Context manager for recording events for the entire application duration.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the entire application (StandardSpanName.APPLICATION).
    """
    span = on_app_start()

    try:
        yield span
    finally:
        on_app_end()


@contextmanager
def model_init() -> Generator[Span, None, None]:
    """Context manager for recording events for the model initialization.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the model initialization (StandardTrainingJobSpanName.MODEL_INIT).
    """
    span = on_model_init_start()
    try:
        yield span
    finally:
        on_model_init_end()


@contextmanager
def dataloader_init() -> Generator[Span, None, None]:
    """Context manager for recording events for the dataloader initialization.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the dataloader initialization (StandardTrainingJobSpanName.DATALOADER_INIT).
    """
    span = on_dataloader_init_start()
    try:
        yield span
    finally:
        on_dataloader_init_end()


@contextmanager
def optimizer_init() -> Generator[Span, None, None]:
    """Context manager for recording events for the optimizer initialization.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the optimizer initialization (StandardTrainingJobSpanName.OPTIMIZER_INIT).
    """
    span = on_optimizer_init_start()
    try:
        yield span
    finally:
        on_optimizer_init_end()


@contextmanager
def training_loop(
    train_iterations_start: int,
    train_samples_start: Optional[int] = None,
    train_iterations_target_or_fn: Optional[Union[int, Callable[[], int]]] = None,
    train_samples_target_or_fn: Optional[Union[int, Callable[[], int]]] = None,
) -> Generator[Span, None, None]:
    """Context manager for recording events for the training loop.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

        train_iterations_start (int): The starting iteration number (could be non-zero if the job loads a checkpoint and starts from there).
        train_samples_start (Optional[int]): The starting sample number (could be non-zero if the job loads a checkpoint and starts from there).
            If not provided, the starting sample number will be calculated as `train_iterations_start * global_batch_size`.
        train_iterations_target (Optional[int]): Target number of training iterations or callable to generate it.
            If not provided, the target will be fetched from the config.
        train_samples_target (Optional[int]): Target number of training samples or function to generate the number.
            If not provided, the target will be fetched from the config.

        NOTE: If the throughput logging is enabled, the target number of training iterations and samples must be provided
        either in the config or via this callback.
    Yields:
        the span corresponding to the training loop (StandardTrainingJobSpanName.TRAINING_LOOP).
    """
    span = on_train_start(
        train_iterations_start=train_iterations_start,
        train_samples_start=train_samples_start,
        train_iterations_target_or_fn=train_iterations_target_or_fn,
        train_samples_target_or_fn=train_samples_target_or_fn,
    )

    try:
        yield span
    finally:
        on_train_end()


@contextmanager
def training_iteration() -> Generator[Span, None, None]:
    """Context manager for recording events for a single training iteration.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the training batch (StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION).
    """
    span = on_training_single_iteration_start()

    try:
        yield span
    finally:
        on_training_single_iteration_end()


@contextmanager
def validation_loop() -> Generator[Span, None, None]:
    """Context manager for recording events for the validation loop.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the training loop (StandardTrainingJobSpanName.VALIDATION_LOOP).
    """
    span = on_validation_start()

    try:
        yield span
    finally:
        on_validation_end()


@contextmanager
def validation_iteration() -> Generator[Span, None, None]:
    """Context manager for recording events for a single validation iteration.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the validation batch (StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION).
    """
    span = on_validation_single_iteration_start()
    try:
        yield span
    finally:
        on_validation_single_iteration_end()


@contextmanager
def testing_loop() -> Generator[Span, None, None]:
    """Context manager for recording events for the testing loop.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the testing loop (StandardTrainingJobSpanName.TESTING_LOOP).
    """
    span = on_testing_start()
    try:
        yield span
    finally:
        on_testing_end()


@contextmanager
def checkpoint_load() -> Generator[Span, None, None]:
    """Context manager for recording events for the checkpoint loading.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.

    Yields:
        the span corresponding to the checkpoint loading (StandardTrainingJobSpanName.CHECKPOINT_LOAD).
    """
    span = on_load_checkpoint_start()
    try:
        yield span
    finally:
        on_load_checkpoint_end()


@contextmanager
def checkpoint_save(global_step: int) -> Generator[Span, None, None]:
    """Context manager for recording events for the synchronous checkpoint saving.

    Note that:
    - This context manager is a no-op when one_logger telemetry is disabled
    - It handles errors in telemetry code according to the according to the configured error handling strategy.


    Args:
        global_step (int): The global step (number of completed training iterations) at the time the checkpoint is saved.

    Yields:
        the span corresponding to the synchronous checkpoint saving
        (StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC or StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC)
        depending on the checkpoint strategy set in TrainingTelemetryConfig.save_checkpoint_strategy.
    """
    span = on_save_checkpoint_start(global_step)
    try:
        yield span
    except Exception:
        raise
    else:
        training_config = TrainingTelemetryProvider.instance().config.telemetry_config
        if training_config and training_config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            on_save_checkpoint_success(global_step)
    finally:
        on_save_checkpoint_end()
