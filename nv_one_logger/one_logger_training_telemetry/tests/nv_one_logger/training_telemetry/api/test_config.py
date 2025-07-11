# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TrainingTelemetryConfig class.

This module contains tests that verify the functionality of the TrainingTelemetryConfig class,
including initialization, validation, and handling of various configuration parameters.
"""

import pytest
from nv_one_logger.api.config import ApplicationType
from nv_one_logger.core.exceptions import OneLoggerError

from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig


def test_basic_config_initialization() -> None:
    """Test that a basic config can be initialized with required fields."""
    config = TrainingTelemetryConfig(
        world_size_or_fn=4,
        global_batch_size_or_fn=32,
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
    )
    assert config.world_size == 4
    assert config.global_batch_size == 32
    assert config.app_type == ApplicationType.TRAINING
    assert config.enable_for_current_rank is False
    assert config.log_every_n_train_iterations == 50


def test_config_with_callable_values() -> None:
    """Test that config can be initialized with callable values."""

    def get_world_size() -> int:
        return 8

    def get_batch_size() -> int:
        return 64

    config = TrainingTelemetryConfig(
        world_size_or_fn=get_world_size,
        global_batch_size_or_fn=get_batch_size,
        application_name="test_app",
        perf_tag_or_fn=lambda: "test_perf",
        session_tag_or_fn=lambda: "test_session",
        app_type_or_fn=lambda: ApplicationType.TRAINING,
        is_baseline_run_or_fn=lambda: False,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
    )
    assert config.world_size == 8
    assert config.global_batch_size == 64
    assert config.perf_tag == "test_perf"
    assert config.session_tag == "test_session"
    assert config.app_type == ApplicationType.TRAINING
    assert config.is_baseline_run is False


def test_invalid_world_size() -> None:
    """Test that initialization fails with invalid world_size."""
    with pytest.raises(OneLoggerError, match="world_size must be set to a non-zero value"):
        TrainingTelemetryConfig(
            world_size_or_fn=0,
            global_batch_size_or_fn=32,
            application_name="test_app",
            perf_tag_or_fn="test_perf",
            session_tag_or_fn="test_session",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        )


def test_invalid_global_batch_size() -> None:
    """Test that initialization fails with invalid global_batch_size."""
    with pytest.raises(OneLoggerError, match="global_batch_size must be set to a non-zero value"):
        TrainingTelemetryConfig(
            world_size_or_fn=4,
            global_batch_size_or_fn=0,
            application_name="test_app",
            perf_tag_or_fn="test_perf",
            session_tag_or_fn="test_session",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        )


def test_throughput_logging_validation() -> None:
    """Test validation of throughput logging related fields."""
    with pytest.raises(OneLoggerError, match="flops_per_sample must be set to a positive value when is_log_throughput_enabled is True"):
        TrainingTelemetryConfig(
            world_size_or_fn=4,
            global_batch_size_or_fn=32,
            is_log_throughput_enabled_or_fn=True,
            application_name="test_app",
            perf_tag_or_fn="test_perf",
            session_tag_or_fn="test_session",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=10000,
        )

    # valid config with throughput logging
    config = TrainingTelemetryConfig(
        world_size_or_fn=4,
        global_batch_size_or_fn=32,
        is_log_throughput_enabled_or_fn=True,
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
        train_iterations_target_or_fn=1000,
        train_samples_target_or_fn=10000,
        flops_per_sample_or_fn=5,
    )
    assert config.is_log_throughput_enabled is True
    assert config.flops_per_sample == 5

    # valid config with throughput logging disabled
    config = TrainingTelemetryConfig(
        world_size_or_fn=4,
        global_batch_size_or_fn=32,
        is_log_throughput_enabled_or_fn=False,
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
    )
    assert config.is_log_throughput_enabled is False
