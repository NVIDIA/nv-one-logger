# SPDX-License-Identifier: Apache-2.0
# This is a simple example of how to use the training telemetry library.
# The code here is meant for demonstration purposes; you cannot run this code as
# the Python project doesn't have the needed dependencies for this example (in particular, pytorch).

# flake8: noqa
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from nv_one_logger.api.timed_span import timed_span
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.exporter.file_exporter import FileExporter
from torch.utils.data import DataLoader, TensorDataset

from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.context import application, checkpoint_save, training_iteration, training_loop
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

# Generate some random data for this example
torch.manual_seed(42)
# Generate random input features and binary labels
X = torch.randn(1000, 10, dtype=torch.float32)
y = (X.sum(dim=1) > 0).float()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
num_epochs = 10

# Initialize the telemetry provider with a default configuration
config = TrainingTelemetryConfig(
    application_name="test_app",
    session_tag_or_fn="test_session",
    is_log_throughput_enabled_or_fn=True,
    save_checkpoint_strategy=CheckPointStrategy.SYNC,
    training_loop_config=TrainingLoopConfig(
        perf_tag_or_fn="test_perf",
        log_every_n_train_iterations=10,
        world_size_or_fn=5,
        flops_per_sample_or_fn=100,
        global_batch_size_or_fn=32,
    ),
)
TrainingTelemetryProvider.instance().with_base_telemetry_config(config).with_exporter(
    FileExporter(file_path=Path("training_telemetry.json"))
).configure_provider()


class SimpleModel(nn.Module):
    """A simple model to be used in this example."""

    def __init__(self) -> None:
        """Initialize the SimpleModel class."""
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Any:
        """Forward path."""
        return self.layers(x)


@application()
def main() -> None:
    """Run the application."""
    # Initialize model, loss function and optimizer
    model = SimpleModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    with training_loop(train_iterations_start=0) as training_span:
        current_iteration = 0
        accuracy = torch.tensor(float("nan"))
        loss = torch.tensor(float("nan"))

        # The library and the context managers collect certain attributes (metrics) automatically.
        # But you can also add custom attributes or fire custom events for any span.
        training_span.add_attribute("my_custom_attribute", "my_custom_value")

        for epoch in range(num_epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                with training_iteration():
                    # Forward pass
                    # Context manager such as @application or @training_iteration help you tell the library
                    # where each span starts and ends. But you can also use the timed_span API to do the same
                    # for standards spans (the ones from StandardTrainingJobSpanName) that do not have a
                    # dedicated context manager or if you want to define your own custom spans.
                    with timed_span(StandardTrainingJobSpanName.MODEL_FORWARD):
                        outputs = model(inputs)
                        with timed_span("loss_calculation_span", span_attributes=Attributes({"my_custom_attribute": "my_custom_value"})):
                            loss = criterion(outputs.squeeze(), targets)

                    # Backward pass and optimize
                    with timed_span(StandardTrainingJobSpanName.ZERO_GRAD):
                        optimizer.zero_grad()
                    with timed_span(StandardTrainingJobSpanName.MODEL_BACKWARD):
                        loss.backward()
                    with timed_span(StandardTrainingJobSpanName.OPTIMIZER_UPDATE):
                        optimizer.step()

                    # Calculate accuracy
                    predictions = (outputs.squeeze() > 0.5).float()
                    accuracy = (predictions == targets).float().mean()

                    current_iteration += 1

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {accuracy.item():.4f}"
                )
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "iteration": current_iteration,
                    "loss": loss.item(),
                    "accuracy": accuracy.item(),
                }
                with checkpoint_save(global_step=current_iteration):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        checkpoint_file_name = os.path.join(temp_dir, f"checkpoint_iter_{current_iteration}.pt")
                        torch.save(checkpoint, checkpoint_file_name)


if __name__ == "__main__":
    main()
