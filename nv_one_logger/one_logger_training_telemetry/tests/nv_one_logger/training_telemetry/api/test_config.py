# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TrainingTelemetryConfig class.

This module contains tests that verify the functionality of the TrainingTelemetryConfig class,
which contains training-specific configuration options.
"""

import pytest

from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig


def test_basic_config_initialization() -> None:
    """Test basic TrainingTelemetryConfig initialization."""
    config = TrainingTelemetryConfig(
        global_batch_size_or_fn=32,
        perf_tag_or_fn="test_perf",
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
    )
    assert config.global_batch_size == 32
    assert config.perf_tag == "test_perf"
    assert config.log_every_n_train_iterations == 50
    assert config.app_type == "training"  # Default from TelemetryConfig


def test_config_with_callable_values() -> None:
    """Test that config can be initialized with callable values."""

    def get_batch_size() -> int:
        return 64

    config = TrainingTelemetryConfig(
        global_batch_size_or_fn=get_batch_size,
        perf_tag_or_fn=lambda: "test_perf",
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
    )
    assert config.global_batch_size == 64
    assert config.perf_tag == "test_perf"


def test_invalid_global_batch_size() -> None:
    """Test that initialization fails with invalid global_batch_size."""
    with pytest.raises(OneLoggerError, match="global_batch_size must be set to a positive value"):
        TrainingTelemetryConfig(
            global_batch_size_or_fn=0,
            perf_tag_or_fn="test_perf",
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        )


def test_throughput_logging_validation() -> None:
    """Test validation of throughput logging related fields."""
    with pytest.raises(OneLoggerError, match="flops_per_sample must be set to a positive value when is_log_throughput_enabled is True"):
        TrainingTelemetryConfig(
            global_batch_size_or_fn=32,
            perf_tag_or_fn="test_perf",
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
            is_log_throughput_enabled_or_fn=True,
            train_iterations_target_or_fn=1000,
            train_samples_target_or_fn=10000,
        )


def test_valid_throughput_logging_config() -> None:
    """Test valid throughput logging configuration."""
    config = TrainingTelemetryConfig(
        global_batch_size_or_fn=32,
        perf_tag_or_fn="test_perf",
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
        is_log_throughput_enabled_or_fn=True,
        flops_per_sample_or_fn=100,
        train_iterations_target_or_fn=1000,
        train_samples_target_or_fn=10000,
    )
    assert config.is_log_throughput_enabled is True
    assert config.flops_per_sample == 100
    assert config.train_iterations_target == 1000
    assert config.train_samples_target == 10000


def test_optional_fields() -> None:
    """Test optional fields in TrainingTelemetryConfig."""
    config = TrainingTelemetryConfig(
        global_batch_size_or_fn=32,
        perf_tag_or_fn="test_perf",
        micro_batch_size_or_fn=16,
        seq_length_or_fn=512,
    )
    assert config.micro_batch_size == 16
    assert config.seq_length == 512
    assert config.flops_per_sample is None  # Not set
    assert config.train_iterations_target is None  # Not set
