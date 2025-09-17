# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.utils import patch_method
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
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from overrides import override

from nv_one_logger.training_telemetry.integration._compat.lightning import (
    STEP_OUTPUT,
    Callback,
    LightningModule,
    Trainer,
)


################################################################################################################################
class TimeEventCallback(Callback):
    """A custom Pytorch Lightning Callback class that calls the appropriate training telemetry callbacks for various training events.

    To enable telemetry, you can simply add this callback to the callbacks list of the Trainer. Since the
    on_load_checkpoint and on_save_checkpoint() hooks of the
    Callback interface are called before the checkpoints are loaded/ saved, telemetry cannot measure the duration of checkpointing.
    Therefore, if is_save_checkpoint_enabled is True, you must use the OneLoggerPTLTrainer class instead. Even if you do not
    plan to collect telemetry on checkpointing, we recommend using the OneLoggerPTLTrainer class.
    """

    def __init__(self, training_telemetry_provider: TrainingTelemetryProvider, call_on_app_start: bool = True):
        """Initialize the TimeEventCallback.

        Args:
            training_telemetry_provider (TrainingTelemetryProvider): The training telemetry provider.
            call_on_app_start (bool): Whether to call on_app_start() during initialization. Defaults to True. Set to False when
                you need explicit lifecycle control to avoid an implicit on_app_start call.
        """
        self._provider: TrainingTelemetryProvider = training_telemetry_provider
        if call_on_app_start:
            on_app_start()

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute when the train begins."""
        # Get the training config from the provider
        training_config = self._provider.config.telemetry_config
        if training_config is None:
            raise OneLoggerError(
                "Training telemetry config must be set before the start of training. "
                "See the documentation for TrainingTelemetryProvider.set_training_telemetry_config for more details."
            )
        global_batch_size = training_config.global_batch_size
        on_train_start(train_iterations_start=trainer.global_step, train_samples_start=trainer.global_step * global_batch_size)

    @override
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute when the train ends."""
        on_train_end()

    @override
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Execute when the train batch begins."""
        on_training_single_iteration_start()

    @override
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Execute when the train batch ends."""
        on_training_single_iteration_end()

    @override
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute when the validation loop begins."""
        on_validation_start()

    @override
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute when the validation loop ends."""
        # TODO(bqi): This is safety net logic as PTL bug is not fixed yet.
        # PTL bug: https://github.com/Lightning-AI/pytorch-lightning/issues/20999
        active_spans = TrainingTelemetryProvider.instance().recorder.get_active_spans_by_name(StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION)
        if OneLoggerProvider.instance().one_logger_enabled and len(active_spans) > 0:
            on_validation_single_iteration_end()
        on_validation_end()

    @override
    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Execute when the validation batch begins."""
        # TODO(bqi): This is safety net logic as PTL bug is not fixed yet.
        # PTL bug: https://github.com/Lightning-AI/pytorch-lightning/issues/20999
        active_spans = TrainingTelemetryProvider.instance().recorder.get_active_spans_by_name(StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION)
        if OneLoggerProvider.instance().one_logger_enabled and len(active_spans) > 0:
            on_validation_single_iteration_end()
        on_validation_single_iteration_start()

    @override
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Execute when the validation batch ends."""
        on_validation_single_iteration_end()

    # The following hooks are not part of the Callback interface, but the library calls them automatically (no need for app code to call them explicitly).
    def on_save_checkpoint_start(self, global_step: int) -> None:
        """Execute when the checkpoint save starts."""
        on_save_checkpoint_start(global_step)

    def on_save_checkpoint_success(self, global_step: int) -> None:
        """Execute when the checkpoint save is successful."""
        on_save_checkpoint_success(global_step=global_step)

    def on_save_checkpoint_end(self, global_step: Optional[int] = None) -> None:
        """Execute when the checkpoint save ends."""
        on_save_checkpoint_end()

    # The following hooks are not part of the Callback interface, but we add them here to make it easier to add telemetry for these events.
    # This means they need to be explicitly called in the application (as opposed to being called automatically by the lightning framework).
    def on_app_end(self) -> None:
        """Execute when the application ends."""
        on_app_end()

    def on_model_init_start(self) -> None:
        """Execute when the model initialization starts."""
        on_model_init_start()

    def on_model_init_end(self) -> None:
        """Execute when the model initialization ends."""
        on_model_init_end()

    def on_dataloader_init_start(self) -> None:
        """Execute when the dataloader initialization starts."""
        on_dataloader_init_start()

    def on_dataloader_init_end(self) -> None:
        """Execute when the dataloader initialization ends."""
        on_dataloader_init_end()

    def on_optimizer_init_start(self) -> None:
        """Execute when the optimizer initialization starts."""
        on_optimizer_init_start()

    def on_optimizer_init_end(self) -> None:
        """Execute when the optimizer initialization ends."""
        on_optimizer_init_end()

    def on_load_checkpoint_start(self) -> None:
        """Execute when the checkpoint loading starts."""
        on_load_checkpoint_start()

    def on_load_checkpoint_end(self) -> None:
        """Execute when the checkpoint loading ends."""
        on_load_checkpoint_end()

    def on_testing_start(self) -> None:
        """Execute when the testing starts."""
        on_testing_start()

    def on_testing_end(self) -> None:
        """Execute when the testing ends."""
        on_testing_end()


def hook_trainer_cls(
    cls: Type[Trainer], training_telemetry_provider: TrainingTelemetryProvider, telemetry_callback: Optional[TimeEventCallback] = None
) -> Tuple[Type[Trainer], TimeEventCallback]:
    """Wrap certain methods of the trainer class to add telemetry hooks.

    Note: In PyTorch Lightning (PTL), this only intercepts the synchronous
    save_checkpoint path; asynchronous checkpointing is not covered by this hook.

    Args:
        cls: The trainer class to hook.
        training_telemetry_provider (TrainingTelemetryProvider): The training telemetry provider.

    Returns:
        A tuple containing:
        - The Trainer class with the following additions:
            - telemetry callback added to the callbacks list
            - a modified save_checkpoint method.
            - a new read-only property called  "nv_one_logger_callback" that contains the telemetry callback.
            You can use this class as a drop-in replacement for the Trainer class.
        - The telemetry callback instance.
    """
    # Create the callback instance if needed
    if telemetry_callback is None:
        telemetry_callback = TimeEventCallback(training_telemetry_provider)

    # patch the constructor to add the telemetry callback to the callbacks list
    def wrapped_init(original_init: Callable[..., Any], self: Trainer, *args: Any, **kwargs: Any) -> Any:
        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, list):
            raise ValueError("The 'callbacks' argument must be a list.")

        # Add time_event_callback to the callbacks list
        callbacks = [telemetry_callback] + callbacks
        kwargs["callbacks"] = callbacks

        # Call the original __init__ method
        original_init(self, *args, **kwargs)
        self._nv_one_logger_callback = telemetry_callback

    cls.__init__ = patch_method(cls.__init__)(wrapped_init)

    def getter(self: Any) -> Any:
        return getattr(self, "_nv_one_logger_callback", None)

    setattr(cls, "nv_one_logger_callback", property(getter))  # noqa: B010

    # patch the save_checkpoint method to call the appropriate training telemetry callbacks
    def wrapped_saved_checkpoint(original_save_checkpoint_method: Callable[..., Any], self: Trainer, *args: Any, **kwargs: Any) -> Any:
        telemetry_callback.on_save_checkpoint_start(global_step=self.global_step)

        # Call the original method
        result = original_save_checkpoint_method(self, *args, **kwargs)

        telemetry_callback.on_save_checkpoint_success(global_step=self.global_step)
        telemetry_callback.on_save_checkpoint_end()
        return result

    cls.save_checkpoint = patch_method(cls.save_checkpoint)(wrapped_saved_checkpoint)

    return cls, telemetry_callback


class OneLoggerPTLTrainer(Trainer):
    """Pytorch Lightning(PTL) Trainer with training telemetry integration.

    This custom PTL Trainer is a drop-in replacement for ptl.Trainer. It automatically adds a custom callback to the trainer
    that calls the appropriate training telemetry callbacks for various training events.

    Since PyTorch Lightning Callback class does not provide  "after checkpoint save" hooks,
    we created this custom PTL Trainer class to override save_checkpoint method.
    """

    def __init__(self, trainer_config: Dict[str, Any], training_telemetry_provider: TrainingTelemetryProvider):
        """Initialize the OneLoggerPTLTrainer.

        Args:
            trainer_config (Dict[str, Any]): The configuration for the PyTorch Lightning Trainer.
            training_telemetry_provider (TrainingTelemetryProvider): The training telemetry provider.
        """
        self._nv_one_logger_callback = TimeEventCallback(training_telemetry_provider)
        callbacks = [self._nv_one_logger_callback] + trainer_config.get("callbacks", [])
        trainer_config["callbacks"] = callbacks

        super().__init__(**trainer_config)

    @override
    def save_checkpoint(self, filepath: Union[str, Path], weights_only: bool = False, storage_options: Optional[Any] = None) -> None:
        """Save a model checkpoint and call the appropriate training telemetry callbacks.

        Args:
            filepath (str): The file path where the checkpoint will be
                saved.
            weights_only (bool, optional): If True, only the model's
                weights are saved. Defaults to False.
            storage_options (dict, optional): Additional storage
                options. Defaults to None.
        """
        self._nv_one_logger_callback.on_save_checkpoint_start(global_step=self.global_step)
        try:
            # Prefer modern signature if available
            super().save_checkpoint(filepath, weights_only, storage_options)
        except TypeError:
            # Fallback for older Lightning versions without storage_options
            super().save_checkpoint(filepath, weights_only)
        self._nv_one_logger_callback.on_save_checkpoint_success(global_step=self.global_step)
        self._nv_one_logger_callback.on_save_checkpoint_end()

    @property
    def nv_one_logger_callback(self) -> TimeEventCallback:
        """Get the TimeEventCallback instance.

        You can use the TimeEventCallback instance to invoke the training telemetry callbacks that
        are not called automatically (i.e., are not part of the Lightning Callback interface).
        See README.md for more details.
        """
        return self._nv_one_logger_callback
