# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
import pytest
from nv_one_logger.api.config import ApplicationType, OneLoggerErrorHandlingStrategy
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig


@pytest.fixture
def config() -> TrainingTelemetryConfig:
    """Create a configuration for Training Telemetry."""
    training_telemetry_config = TrainingTelemetryConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        custom_metadata={"test_metadata": "test_metadata_value"},
        error_handling_strategy=OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS,
        enable_one_logger=True,
        enable_for_current_rank=True,
        is_train_iterations_enabled_or_fn=True,
        is_test_iterations_enabled_or_fn=True,
        is_save_checkpoint_enabled_or_fn=True,
        is_log_throughput_enabled_or_fn=True,
        summary_data_schema_version_or_fn="1.0",
        training_loop_config=TrainingLoopConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=10,
            global_batch_size_or_fn=32,
            log_every_n_train_iterations=5,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=10000,
            micro_batch_size_or_fn=4,
            seq_length_or_fn=512,
            flops_per_sample_or_fn=100,
        ),
    )
    return training_telemetry_config
