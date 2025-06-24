# SPDX-License-Identifier: Apache-2.0
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
from nv_one_logger.exporter.exporter import Exporter

from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig


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
        world_size_or_fn=10,
        is_log_throughput_enabled_or_fn=True,
        flops_per_sample_or_fn=100,
        global_batch_size_or_fn=32,
        log_every_n_train_iterations=10,
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_session",
        enable_one_logger=True,
    )
    config.validate_config()
    return config


@pytest.fixture
def mock_exporter() -> Generator[Exporter, None, None]:
    """Fixture that sets up a mock exporter."""
    exporter = MagicMock(spec=Exporter)

    yield exporter

    exporter.reset_mock()
