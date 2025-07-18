# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
import uuid
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
from nv_one_logger.api.config import ApplicationType, OneLoggerErrorHandlingStrategy
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.internal.singleton import SingletonMeta
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.wandb.exporter.wandb_exporter import Config as WandBConfig

from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import (
    V1CompatibleExporter,
    V1CompatibleWandbExporterAsync,
    V1CompatibleWandbExporterSync,
)


@pytest.fixture(autouse=True, scope="function")
def reset_v2_provider():
    """Reset the v2 provider singleton to isolate tests."""
    OneLoggerProvider.instance()._config = None  # type: ignore
    OneLoggerProvider.instance()._recorder = None  # type: ignore


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset the state of the singletons."""
    with SingletonMeta._lock:
        SingletonMeta._instances.pop(TrainingTelemetryProvider, None)
        SingletonMeta._instances.pop(OneLoggerProvider, None)


class TestV1CompatibleExporter:
    """Test cases for the V1CompatibleExporter factory class."""

    def test_factory_creates_sync_exporter(self, config: TrainingTelemetryConfig) -> None:
        """Test that factory creates V1CompatibleWandbExporterSync when async_mode=False."""
        exporter = V1CompatibleExporter(config, async_mode=False)
        
        assert exporter.is_async is False
        assert isinstance(exporter.exporter, V1CompatibleWandbExporterSync)

    def test_factory_creates_async_exporter(self, config: TrainingTelemetryConfig) -> None:
        """Test that factory creates V1CompatibleWandbExporterAsync when async_mode=True."""
        exporter = V1CompatibleExporter(config, async_mode=True)
        
        assert exporter.is_async is True
        assert isinstance(exporter.exporter, V1CompatibleWandbExporterAsync)

    def test_factory_default_creates_sync_exporter(self, config: TrainingTelemetryConfig) -> None:
        """Test that factory creates sync exporter by default."""
        exporter = V1CompatibleExporter(config)
        
        assert exporter.is_async is False
        assert isinstance(exporter.exporter, V1CompatibleWandbExporterSync) 