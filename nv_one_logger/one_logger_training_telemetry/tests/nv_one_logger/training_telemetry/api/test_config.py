# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TrainingTelemetryConfig class.

This module contains tests that verify the functionality of the TrainingTelemetryConfig class,
including initialization, validation, and handling of various configuration parameters.
"""

import pytest
from nv_one_logger.api.config import ApplicationType
from nv_one_logger.core.exceptions import OneLoggerError

from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig


def test_basic_config_initialization() -> None:
    """Test that a basic config can be initialized with required fields."""
    config = TrainingTelemetryConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
        training_loop_config=TrainingLoopConfig(
            world_size_or_fn=4,
            global_batch_size_or_fn=32,
            perf_tag_or_fn="test_perf",
        ),
    )
    assert config.app_type == ApplicationType.TRAINING
    assert config.enable_for_current_rank is False
    assert config.training_loop_config is not None
    assert config.training_loop_config.world_size == 4
    assert config.training_loop_config.global_batch_size == 32
    assert config.training_loop_config.log_every_n_train_iterations == 50


def test_config_with_callable_values() -> None:
    """Test that config can be initialized with callable values."""

    def get_world_size() -> int:
        return 8

    def get_batch_size() -> int:
        return 64

    config = TrainingTelemetryConfig(
        application_name="test_app",
        session_tag_or_fn=lambda: "test_session",
        app_type_or_fn=lambda: ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        training_loop_config=TrainingLoopConfig(
            world_size_or_fn=get_world_size,
            global_batch_size_or_fn=get_batch_size,
            perf_tag_or_fn=lambda: "test_perf",
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        ),
    )
    assert config.training_loop_config is not None
    assert config.training_loop_config.world_size == 8
    assert config.training_loop_config.global_batch_size == 64
    assert config.training_loop_config.perf_tag == "test_perf"
    assert config.session_tag == "test_session"
    assert config.app_type == ApplicationType.TRAINING
    assert config.is_baseline_run is False


def test_invalid_world_size() -> None:
    """Test that initialization fails with invalid world_size."""
    with pytest.raises(OneLoggerError, match="world_size must be set to a non-zero value"):
        TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            training_loop_config=TrainingLoopConfig(
                world_size_or_fn=0,
                global_batch_size_or_fn=32,
                perf_tag_or_fn="test_perf",
                save_checkpoint_strategy=CheckPointStrategy.SYNC,
            ),
        )


def test_invalid_global_batch_size() -> None:
    """Test that initialization fails with invalid global_batch_size."""
    with pytest.raises(OneLoggerError, match="global_batch_size must be set to a non-zero value"):
        TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            training_loop_config=TrainingLoopConfig(
                world_size_or_fn=4,
                global_batch_size_or_fn=0,
                perf_tag_or_fn="test_perf",
                save_checkpoint_strategy=CheckPointStrategy.SYNC,
            ),
        )


def test_throughput_logging_validation() -> None:
    """Test validation of throughput logging related fields."""
    with pytest.raises(OneLoggerError, match="flops_per_sample must be set to a positive value when is_log_throughput_enabled is True"):
        TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                world_size_or_fn=4,
                global_batch_size_or_fn=32,
                perf_tag_or_fn="test_perf",
                save_checkpoint_strategy=CheckPointStrategy.SYNC,
                train_iterations_target_or_fn=1000,
                train_samples_target_or_fn=10000,
            ),
        )

    # valid config with throughput logging
    config = TrainingTelemetryConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        is_log_throughput_enabled_or_fn=True,
        training_loop_config=TrainingLoopConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=4,
            global_batch_size_or_fn=32,
            flops_per_sample_or_fn=5,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=10000,
        ),
    )
    assert config.is_log_throughput_enabled is True
    assert config.training_loop_config is not None
    assert config.training_loop_config.flops_per_sample == 5

    # valid config with throughput logging disabled
    config = TrainingTelemetryConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        is_log_throughput_enabled_or_fn=False,
        training_loop_config=TrainingLoopConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=4,
            global_batch_size_or_fn=32,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        ),
    )
    assert config.is_log_throughput_enabled is False
