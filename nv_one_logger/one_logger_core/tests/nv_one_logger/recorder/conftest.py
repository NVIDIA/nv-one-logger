from typing import Generator
from unittest.mock import MagicMock

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.exporter.exporter import Exporter


@pytest.fixture
def config() -> OneLoggerConfig:
    """Create a configuration for OneLogger."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
    )

    return config


@pytest.fixture
def mock_exporter() -> Generator[Exporter, None, None]:
    """Fixture that sets up a mock exporter."""
    exporter = MagicMock(spec=Exporter)

    yield exporter

    exporter.reset_mock()
