# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
import pytest
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig


@pytest.fixture
def config() -> TrainingTelemetryConfig:
    """Create a configuration for Training Telemetry."""
    training_telemetry_config = TrainingTelemetryConfig(
        world_size_or_fn=10,
        global_batch_size_or_fn=32,
        perf_tag_or_fn="test_perf",
        log_every_n_train_iterations=5,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
        train_iterations_target_or_fn=1000,
        train_samples_target_or_fn=10000,
        micro_batch_size_or_fn=4,
        seq_length_or_fn=512,
        flops_per_sample_or_fn=100,
        is_train_iterations_enabled_or_fn=True,
        is_test_iterations_enabled_or_fn=True,
        is_save_checkpoint_enabled_or_fn=True,
        is_log_throughput_enabled_or_fn=True,
    )
    return training_telemetry_config
