# SPDX-License-Identifier: Apache-2.0
from typing import Generator, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from .utils import reset_singletong_providers_for_test


def configure_provider_for_test(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
    """Reset the state of the provider singleton for testing purposes."""
    # Do NOT change this to a fixture that gets called for all tests.
    # Some tests in this module create their own instances of Recorder. Calling this
    # function for those tests interferes with testing the Recorder in isolation.
    reset_singletong_providers_for_test()

    # Create a base OneLoggerConfig with the TrainingTelemetryConfig nested inside
    base_config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        world_size_or_fn=4,
        telemetry_config=config,
    )

    (TrainingTelemetryProvider.instance().with_base_config(base_config).with_exporter(mock_exporter).configure_provider())


def reconfigure_provider(config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
    """Use this function when you want to reconfigure the provider after it is already initialized."""
    cast(MagicMock, mock_exporter).reset_mock()
    configure_provider_for_test(config, mock_exporter)


@pytest.fixture
def mock_time() -> Generator[Mock, None, None]:
    """Patch time.time and provide the corresponding mock."""
    with patch("time.time") as mock_time:
        mock_time.return_value = 0
        yield mock_time


@pytest.fixture
def mock_perf_counter() -> Generator[Mock, None, None]:
    """Patch time.perf_counter and provide the corresponding mock."""
    with patch("time.perf_counter") as mock_perf_counter:
        mock_perf_counter.return_value = 0
        yield mock_perf_counter


@pytest.fixture
def config() -> TrainingTelemetryConfig:
    """Create a configuration for Training Telemetry."""
    config = TrainingTelemetryConfig(
        perf_tag_or_fn="test_perf",
        global_batch_size_or_fn=32,
        flops_per_sample_or_fn=100,
        log_every_n_train_iterations=10,
        train_iterations_target_or_fn=100,
        train_samples_target_or_fn=3200,
        is_log_throughput_enabled_or_fn=True,
    )
    return config


@pytest.fixture
def one_logger_config() -> OneLoggerConfig:
    """Create a OneLoggerConfig with TrainingTelemetryConfig for testing."""
    return OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        is_baseline_run_or_fn=False,
        enable_for_current_rank=True,
        world_size_or_fn=4,
        telemetry_config=TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            global_batch_size_or_fn=32,
            flops_per_sample_or_fn=100,
            log_every_n_train_iterations=10,
            train_iterations_target_or_fn=100,
            train_samples_target_or_fn=3200,
            is_log_throughput_enabled_or_fn=True,
        ),
    )


@pytest.fixture
def mock_exporter() -> Generator[Exporter, None, None]:
    """Fixture that sets up a mock exporter."""
    exporter = MagicMock(spec=Exporter)

    yield exporter

    exporter.reset_mock()


@pytest.fixture
def training_telemetry_provider(valid_config, mock_exporter) -> TrainingTelemetryProvider:
    """Fixture that returns a configured TrainingTelemetryProvider instance."""
    provider = TrainingTelemetryProvider.instance()
    provider.with_base_config(valid_config).with_exporter(mock_exporter).configure_provider()
    return provider
