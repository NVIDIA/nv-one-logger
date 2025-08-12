from typing import Dict, Generator, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.api.telemetry_config import ApplicationType
from nv_one_logger.core.attributes import AttributeValue
from nv_one_logger.exporter.exporter import Exporter


@pytest.fixture()
def mock_time() -> Generator[Mock, None, None]:
    """Patch time.time and provide the corresponding mock."""
    with patch("time.time") as mock_time:
        yield mock_time


@pytest.fixture()
def mock_perf_counter() -> Generator[Mock, None, None]:
    """Patch time.perf_counter and provide the corresponding mock."""
    with patch("time.perf_counter") as mock_perf_counter:
        yield mock_perf_counter


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


class MockTelemetryConfig:
    """Mock implementation of TelemetryConfig protocol for testing."""

    def __init__(
        self,
        app_type: ApplicationType = ApplicationType.TRAINING,
        is_train_iterations_enabled: bool = True,
        is_validation_iterations_enabled: bool = True,
        is_test_iterations_enabled: bool = True,
        is_save_checkpoint_enabled: bool = True,
        custom_metadata: Optional[Dict[str, AttributeValue]] = None,
    ):
        self._app_type = app_type
        self._is_train_iterations_enabled = is_train_iterations_enabled
        self._is_validation_iterations_enabled = is_validation_iterations_enabled
        self._is_test_iterations_enabled = is_test_iterations_enabled
        self._is_save_checkpoint_enabled = is_save_checkpoint_enabled
        self._custom_metadata = custom_metadata

    @property
    def app_type(self) -> ApplicationType:
        """Get the application type."""
        return self._app_type

    @property
    def is_train_iterations_enabled(self) -> bool:
        """Get whether train iterations are enabled."""
        return self._is_train_iterations_enabled

    @property
    def is_validation_iterations_enabled(self) -> bool:
        """Get whether validation iterations are enabled."""
        return self._is_validation_iterations_enabled

    @property
    def is_test_iterations_enabled(self) -> bool:
        """Get whether test iterations are enabled."""
        return self._is_test_iterations_enabled

    @property
    def is_save_checkpoint_enabled(self) -> bool:
        """Get whether save checkpoint is enabled."""
        return self._is_save_checkpoint_enabled

    @property
    def custom_metadata(self) -> Optional[Dict[str, AttributeValue]]:
        """Get the telemetry metadata."""
        return self._custom_metadata


class MockTelemetryConfigWithCallables:
    """Mock implementation of TelemetryConfig protocol with different values for testing callable behavior."""

    def __init__(self):
        self._app_type = ApplicationType.VALIDATION
        self._is_train_iterations_enabled = False

    @property
    def app_type(self) -> ApplicationType:
        """Get the application type."""
        return self._app_type

    @property
    def is_train_iterations_enabled(self) -> bool:
        """Get whether train iterations are enabled."""
        return self._is_train_iterations_enabled

    @property
    def is_validation_iterations_enabled(self) -> bool:
        """Get whether validation iterations are enabled."""
        return True

    @property
    def is_test_iterations_enabled(self) -> bool:
        """Get whether test iterations are enabled."""
        return True

    @property
    def is_save_checkpoint_enabled(self) -> bool:
        """Get whether save checkpoint is enabled."""
        return True

    @property
    def custom_metadata(self) -> Optional[Dict[str, AttributeValue]]:
        """Get the telemetry metadata."""
        return None


class MockTelemetryConfigWithMetadata:
    """Mock implementation of TelemetryConfig protocol with telemetry metadata for testing."""

    def __init__(self, metadata: Optional[Dict[str, AttributeValue]]):
        self._app_type = ApplicationType.TRAINING
        self._is_train_iterations_enabled = True
        self._is_validation_iterations_enabled = True
        self._is_test_iterations_enabled = True
        self._is_save_checkpoint_enabled = True
        self._custom_metadata = metadata

    @property
    def app_type(self) -> ApplicationType:
        """Get the application type."""
        return self._app_type

    @property
    def is_train_iterations_enabled(self) -> bool:
        """Get whether train iterations are enabled."""
        return self._is_train_iterations_enabled

    @property
    def is_validation_iterations_enabled(self) -> bool:
        """Get whether validation iterations are enabled."""
        return self._is_validation_iterations_enabled

    @property
    def is_test_iterations_enabled(self) -> bool:
        """Get whether test iterations are enabled."""
        return self._is_test_iterations_enabled

    @property
    def is_save_checkpoint_enabled(self) -> bool:
        """Get whether save checkpoint is enabled."""
        return self._is_save_checkpoint_enabled

    @property
    def custom_metadata(self) -> Optional[Dict[str, AttributeValue]]:
        """Get the telemetry metadata."""
        return self._custom_metadata


@pytest.fixture
def mock_telemetry_config() -> MockTelemetryConfig:
    """Fixture that provides a basic mock telemetry config."""
    return MockTelemetryConfig()


@pytest.fixture
def mock_telemetry_config_with_callables() -> MockTelemetryConfigWithCallables:
    """Fixture that provides a mock telemetry config with different values for testing callable behavior."""
    return MockTelemetryConfigWithCallables()


@pytest.fixture
def mock_telemetry_config_with_metadata() -> MockTelemetryConfigWithMetadata:
    """Fixture that provides a mock telemetry config with telemetry metadata."""
    custom_metadata: Dict[str, AttributeValue] = {"telemetry_key": "telemetry_value"}
    return MockTelemetryConfigWithMetadata(custom_metadata)
