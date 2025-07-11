from typing import Generator
from unittest.mock import MagicMock

import pytest

from nv_one_logger.api.config import ApplicationType, OneLoggerConfig
from nv_one_logger.exporter.exporter import Exporter


@pytest.fixture
def config() -> OneLoggerConfig:
    """Create a configuration for OneLogger."""
    config = OneLoggerConfig(
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_task",
        app_type_or_fn=ApplicationType.TRAINING,
        enable_one_logger=True,
    )

    return config


@pytest.fixture
def mock_exporter() -> Generator[Exporter, None, None]:
    """Fixture that sets up a mock exporter."""
    exporter = MagicMock(spec=Exporter)

    yield exporter

    exporter.reset_mock()
