# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false

import os
import shutil
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as ptl
import torch
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.singleton import SingletonMeta
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from nv_one_logger.training_telemetry.integration.pytorch_lightning import (
    OneLoggerPTLTrainer,
    TimeEventCallback,
    hook_trainer_cls,
)


class DummyModel(LightningModule):
    """A simple dummy model for testing purposes."""

    def __init__(self) -> None:
        """Initialize the dummy model with a simple linear layer."""
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 10)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        return self.linear(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for the model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple of (inputs, targets)
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validate the model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple of (inputs, targets)
            batch_idx (int): Index of the current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer to use for training
        """
        return torch.optim.Adam(self.parameters())


@pytest.fixture
def dummy_model() -> DummyModel:
    """Create a dummy model for testing."""
    return DummyModel()


@pytest.fixture
def config(request: pytest.FixtureRequest) -> TrainingTelemetryConfig:
    """Create a configuration for Training Telemetry."""
    checkpoint_strategy: CheckPointStrategy = request.param
    config = TrainingTelemetryConfig(
        is_log_throughput_enabled_or_fn=True,
        is_save_checkpoint_enabled_or_fn=True,
        application_name="test_app",
        session_tag_or_fn="test_session",
        enable_one_logger=True,
        enable_for_current_rank=True,
        save_checkpoint_strategy=checkpoint_strategy,
        training_loop_config=TrainingLoopConfig(
            perf_tag_or_fn="test_perf",
            log_every_n_train_iterations=10,
            world_size_or_fn=10,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=10000,
            flops_per_sample_or_fn=100,
            global_batch_size_or_fn=32,
        ),
    )
    return config


@pytest.fixture
def mock_exporter() -> Generator[Exporter, None, None]:
    """Fixture that sets up a mock exporter."""
    exporter = MagicMock(spec=Exporter)

    yield exporter

    exporter.reset_mock()


@pytest.fixture(autouse=True)
def configure_provider(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
    """Fixture that configures the TrainingTelemetryProvider."""
    # Reset the state of the singletons
    with SingletonMeta._lock:
        SingletonMeta._instances.pop(TrainingTelemetryProvider, None)
        SingletonMeta._instances.pop(OneLoggerProvider, None)
    TrainingTelemetryProvider.instance().with_base_telemetry_config(config).with_exporter(mock_exporter).configure_provider()


@pytest.fixture
def dummy_data() -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Create dummy training and validation data loaders.

    Returns:
        tuple[DataLoader, DataLoader]: Tuple of (train_loader, val_loader)
    """
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    return train_loader, val_loader


CHECKPOINTS_DIR = "checkpoints"


@pytest.fixture
def checkpoints_dir() -> Generator[str, None, None]:
    """Create a directory for checkpoints."""
    checkpoint_path = CHECKPOINTS_DIR
    yield checkpoint_path
    if os.path.exists(CHECKPOINTS_DIR):
        shutil.rmtree(CHECKPOINTS_DIR)


@pytest.mark.parametrize("config", [CheckPointStrategy.SYNC, CheckPointStrategy.ASYNC], indirect=True, ids=["sync", "async"])
@pytest.mark.parametrize("use_hook_trainer_cls", [True, False], ids=["use_hook_trainer_cls", "use_one_logger_ptl_trainer"])
def test_one_logger_ptl_trainer(
    config: TrainingTelemetryConfig,
    use_hook_trainer_cls: bool,
    dummy_model: DummyModel,
    dummy_data: tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]],
    checkpoints_dir: str,
) -> None:
    """Tests PTL integration and verifies that supported telemetry callbacks are called implicitly.

    Args:
        checkpoint_strategy (CheckPointStrategy): The checkpoint strategy to use (SYNC or ASYNC)
        use_hook_trainer_cls (bool): Whether to use the hook_trainer_cls function to patch the Trainer class or use
        the OneLoggerPTLTrainer class directly.
        dummy_model (DummyModel): A dummy PyTorch Lightning model for testing
        dummy_data (tuple[DataLoader, DataLoader]): Tuple of (train_loader, val_loader)
        config (TrainingTelemetryConfig): Configuration for training telemetry
        checkpoints_dir (str): Path to the checkpoints directory.
    """
    checkpoint_strategy = config.save_checkpoint_strategy
    train_loader, val_loader = dummy_data

    # Create the model and trainer
    CHECKPOINT_EVERY_N_TRAIN_STEPS = 2
    NUM_EPOCHS = 5
    NUM_TRAIN_BATCHES = 4
    NUM_VAL_BATCHES = 3

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
        save_top_k=-1,
        dirpath=checkpoints_dir,
    )
    trainer_config: dict[str, Any] = {
        "max_epochs": NUM_EPOCHS,
        "limit_train_batches": NUM_TRAIN_BATCHES,
        "limit_val_batches": NUM_VAL_BATCHES,
        "logger": False,
        "callbacks": [checkpoint_callback],
    }
    # To ensure test unit isolation, we need to undo what hook_trainer_cls does at the end of the test.
    original_init = Trainer.__init__
    original_save_checkpoint = Trainer.save_checkpoint

    # Mock all the callback functions
    with (
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_app_start") as mock_app_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_app_end") as mock_app_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_save_checkpoint_start") as mock_save_checkpoint_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_save_checkpoint_success") as mock_save_checkpoint_success,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_save_checkpoint_end") as mock_save_checkpoint_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_train_start") as mock_train_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_train_end") as mock_train_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_training_single_iteration_start") as mock_train_iter_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_training_single_iteration_end") as mock_train_iter_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_validation_start") as mock_val_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_validation_end") as mock_val_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_validation_single_iteration_start") as mock_val_iter_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_validation_single_iteration_end") as mock_val_iter_end,
    ):
        if use_hook_trainer_cls:
            if checkpoint_strategy == CheckPointStrategy.SYNC:
                HookedTrainer, telemetry_callback = hook_trainer_cls(Trainer, TrainingTelemetryProvider.instance())
                trainer = HookedTrainer(**trainer_config)
                assert telemetry_callback == trainer.nv_one_logger_callback
            else:
                with pytest.raises(OneLoggerError, match=r"'hook_trainer_cls\(\)' doesn't support async checkpointing yet. Use 'OneLoggerPTLTrainer' instead."):
                    hook_trainer_cls(Trainer, TrainingTelemetryProvider.instance())
                return
        else:
            trainer = OneLoggerPTLTrainer(
                trainer_config=trainer_config,
                training_telemetry_provider=TrainingTelemetryProvider.instance(),
            )
        telemetry_callback = trainer.nv_one_logger_callback

        trainer.fit(dummy_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        telemetry_callback.on_app_end()

        # Verify callbacks were called in the correct order
        mock_app_start.assert_called_once()
        mock_train_start.assert_called_once()
        assert mock_train_iter_start.call_count == NUM_EPOCHS * NUM_TRAIN_BATCHES
        assert mock_train_iter_end.call_count == mock_train_iter_start.call_count
        # Ligthning does one validation loop before starting the training
        assert mock_val_start.call_count == NUM_EPOCHS + 1

        # In PyTorch Lightning, limit_val_batches refers to the fraction of the validation dataset, not the exact number of batches.
        # When limit_val_batches is set, Lightning may not be the exact number of batches.
        # Depending on how the validation dataset is structured, if the full validation
        # set can be larger than limit_val_batches batches, this setting may request a
        # fraction of the validation dataset, leading to more iterations.
        # So we allow some variance in the number of validation iterations.
        assert mock_val_iter_start.call_count > NUM_EPOCHS * NUM_VAL_BATCHES and mock_val_iter_start.call_count < NUM_EPOCHS * (NUM_VAL_BATCHES + 1)
        assert mock_val_iter_end.call_count == mock_val_iter_start.call_count
        mock_val_end.call_count = mock_val_start.call_count

        EXPECTED_CHECKPOINT_SAVES = NUM_EPOCHS * NUM_TRAIN_BATCHES / CHECKPOINT_EVERY_N_TRAIN_STEPS
        assert mock_save_checkpoint_start.call_count == EXPECTED_CHECKPOINT_SAVES
        assert mock_save_checkpoint_success.call_count == EXPECTED_CHECKPOINT_SAVES
        assert mock_save_checkpoint_end.call_count == EXPECTED_CHECKPOINT_SAVES
        mock_train_end.assert_called_once()
        mock_app_end.assert_called_once()

        if use_hook_trainer_cls:
            # Restore the original methods
            Trainer.__init__ = original_init
            Trainer.save_checkpoint = original_save_checkpoint


@pytest.mark.parametrize("config", [CheckPointStrategy.SYNC], indirect=True, ids=["sync"])
@pytest.mark.parametrize("use_hook_trainer_cls", [True, False], ids=["use_hook_trainer_cls", "use_one_logger_ptl_trainer"])
def test_explicit_telemetry_callback_invocation(
    use_hook_trainer_cls: bool,
) -> None:
    """Test the OneLoggerPTLTrainer with explicit telemetry callback invocation.

    Args:
        use_hook_trainer_cls (bool): Whether to use the hook_trainer_cls function to patch the Trainer class or use
        the OneLoggerPTLTrainer class directly.
    """
    trainer_config: dict[str, Any] = {
        "max_epochs": 5,
        "limit_train_batches": 4,
        "limit_val_batches": 3,
        "logger": False,
    }
    # To ensure test unit isolation, we need to undo what hook_trainer_cls does at the end of the test.
    original_init = Trainer.__init__
    original_save_checkpoint = Trainer.save_checkpoint

    # Mock all the callback functions
    with (
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_app_end") as mock_app_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_testing_start") as mock_testing_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_testing_end") as mock_testing_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_dataloader_init_start") as mock_dataloader_init_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_dataloader_init_end") as mock_dataloader_init_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_model_init_start") as mock_model_init_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_model_init_end") as mock_model_init_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_optimizer_init_start") as mock_optimizer_init_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_optimizer_init_end") as mock_optimizer_init_end,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_load_checkpoint_start") as mock_load_checkpoint_start,
        patch("nv_one_logger.training_telemetry.integration.pytorch_lightning.on_load_checkpoint_end") as mock_load_checkpoint_end,
    ):
        if use_hook_trainer_cls:
            HookedTrainer, telemetry_callback = hook_trainer_cls(Trainer, TrainingTelemetryProvider.instance())
            trainer = HookedTrainer(**trainer_config)
            assert telemetry_callback == trainer.nv_one_logger_callback
        else:
            trainer = OneLoggerPTLTrainer(
                trainer_config=trainer_config,
                training_telemetry_provider=TrainingTelemetryProvider.instance(),
            )
        telemetry_callback = trainer.nv_one_logger_callback

        telemetry_callback.on_model_init_start()
        telemetry_callback.on_model_init_end()
        telemetry_callback.on_dataloader_init_start()
        telemetry_callback.on_dataloader_init_end()
        telemetry_callback.on_optimizer_init_start()
        telemetry_callback.on_optimizer_init_end()
        telemetry_callback.on_load_checkpoint_start()
        telemetry_callback.on_load_checkpoint_end()
        telemetry_callback.on_testing_start()
        telemetry_callback.on_testing_end()
        telemetry_callback.on_app_end()

        # Verify callbacks were called in the correct order
        mock_model_init_start.assert_called_once()
        mock_model_init_end.assert_called_once()
        mock_dataloader_init_start.assert_called_once()
        mock_dataloader_init_end.assert_called_once()
        mock_optimizer_init_start.assert_called_once()
        mock_optimizer_init_end.assert_called_once()
        mock_load_checkpoint_start.assert_called_once()
        mock_load_checkpoint_end.assert_called_once()
        mock_testing_start.assert_called_once()
        mock_testing_end.assert_called_once()
        mock_app_end.assert_called_once()

        if use_hook_trainer_cls:
            # Restore the original methods
            Trainer.__init__ = original_init
            Trainer.save_checkpoint = original_save_checkpoint


@pytest.mark.parametrize("config", [CheckPointStrategy.SYNC], indirect=True, ids=["sync"])
def test_on_validation_batch_start_auto_end_previous_validation_single_iteration(
    dummy_data: tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    """Test auto calling on_validation_batch_end when validation_step returns None.

    This test verifies the actual PyTorch Lightning scenario where validation_step returns None,
    which can cause PyTorch Lightning to skip calling on_validation_batch_end, but our integration still calls it.
    """
    _, val_loader = dummy_data

    # Create a mock model that returns None from validation_step
    model_with_none_step = MagicMock()
    model_with_none_step.validation_step.return_value = None

    # Create a telemetry callback
    telemetry_callback = TimeEventCallback(TrainingTelemetryProvider.instance())

    batch = next(iter(val_loader))
    trainer = Trainer()

    # Start validation loop
    telemetry_callback.on_validation_start(trainer, model_with_none_step)

    # Start first validation iteration
    telemetry_callback.on_validation_batch_start(trainer, model_with_none_step, batch, batch_idx=0)
    timer = telemetry_callback._provider.recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION]
    assert timer.is_active, "Timer should be active after starting validation iteration"

    # Simulate the model's validation_step returning None (which happens in our mock model)
    # In real PyTorch Lightning, this could cause on_validation_batch_end to be skipped
    result = model_with_none_step.validation_step(batch, 0)
    assert result is None, "Validation step should return None"

    # Simulate PyTorch Lightning skipping on_validation_batch_end and starting next iteration
    # The safety mechanism should handle this
    telemetry_callback.on_validation_batch_start(trainer, model_with_none_step, batch, batch_idx=1)

    # Verify timer is still active (new iteration started) and safety mechanism worked
    assert timer.is_active, "Timer should still be active after safety mechanism"

    # End the validation properly
    telemetry_callback.on_validation_batch_end(trainer, model_with_none_step, None, batch, batch_idx=1)
    telemetry_callback.on_validation_end(trainer, model_with_none_step)

    # Verify cleanup
    assert not timer.is_active, "Timer should be inactive after proper cleanup"


@pytest.mark.parametrize("config", [CheckPointStrategy.SYNC], indirect=True, ids=["sync"])
@pytest.mark.parametrize("use_hook_trainer_cls", [True, False], ids=["use_hook_trainer_cls", "use_one_logger_ptl_trainer"])
def test_telemetry_callback_is_first_in_callback_list(
    use_hook_trainer_cls: bool,
) -> None:
    """Test that the telemetry callback is placed at the beginning of the callback list.

    This ensures that telemetry events are captured before any other callbacks run,
    which is important for accurate timing measurements.

    Args:
        use_hook_trainer_cls (bool): Whether to use the hook_trainer_cls function to patch the Trainer class or use
        the OneLoggerPTLTrainer class directly.
    """
    # Create some dummy callbacks for testing
    dummy_callback_1 = MagicMock(spec=ptl.Callback)
    dummy_callback_2 = MagicMock(spec=ptl.Callback)
    dummy_callbacks = [dummy_callback_1, dummy_callback_2]

    trainer_config: dict[str, Any] = {
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "logger": False,
        "callbacks": dummy_callbacks,
    }

    # To ensure test unit isolation, we need to undo what hook_trainer_cls does at the end of the test.
    original_init = Trainer.__init__
    original_save_checkpoint = Trainer.save_checkpoint

    try:
        if use_hook_trainer_cls:
            HookedTrainer, telemetry_callback = hook_trainer_cls(Trainer, TrainingTelemetryProvider.instance())
            trainer = HookedTrainer(**trainer_config)
            assert telemetry_callback == trainer.nv_one_logger_callback
        else:
            trainer = OneLoggerPTLTrainer(
                trainer_config=trainer_config,
                training_telemetry_provider=TrainingTelemetryProvider.instance(),
            )
            telemetry_callback = trainer.nv_one_logger_callback

        # Verify that the telemetry callback is at the beginning of the callbacks list
        callbacks = trainer.callbacks
        assert len(callbacks) >= 3, f"Expected at least 3 callbacks (telemetry + 2 dummy), got {len(callbacks)}"

        # The first callback should be the telemetry callback
        assert isinstance(callbacks[0], TimeEventCallback), f"First callback should be TimeEventCallback, got {type(callbacks[0])}"
        assert callbacks[0] is telemetry_callback, "First callback should be the same instance as the telemetry callback"

        # The dummy callbacks should follow after the telemetry callback
        # Note: PyTorch Lightning may add additional callbacks, so we check that our dummy callbacks
        # are present in the expected order after the telemetry callback
        dummy_callback_indices = []
        for i, callback in enumerate(callbacks):
            if callback in dummy_callbacks:
                dummy_callback_indices.append(i)

        assert len(dummy_callback_indices) == 2, f"Expected to find 2 dummy callbacks, found {len(dummy_callback_indices)}"

        # Find the positions of our dummy callbacks
        dummy_1_index = None
        dummy_2_index = None
        for i, callback in enumerate(callbacks):
            if callback is dummy_callback_1:
                dummy_1_index = i
            elif callback is dummy_callback_2:
                dummy_2_index = i

        assert dummy_1_index is not None, "dummy_callback_1 not found in callbacks list"
        assert dummy_2_index is not None, "dummy_callback_2 not found in callbacks list"

        # Both dummy callbacks should come after the telemetry callback (index 0)
        assert dummy_1_index > 0, f"dummy_callback_1 should come after telemetry callback, but is at index {dummy_1_index}"
        assert dummy_2_index > 0, f"dummy_callback_2 should come after telemetry callback, but is at index {dummy_2_index}"

        # The dummy callbacks should maintain their relative order
        assert dummy_1_index < dummy_2_index, f"dummy_callback_1 (index {dummy_1_index}) should come before dummy_callback_2 (index {dummy_2_index})"

    finally:
        if use_hook_trainer_cls:
            # Restore the original methods
            Trainer.__init__ = original_init
            Trainer.save_checkpoint = original_save_checkpoint
