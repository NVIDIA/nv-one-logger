# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TelemetryConfig protocol."""

from typing import Dict

from nv_one_logger.api.telemetry_config import ApplicationType, TelemetryConfig
from nv_one_logger.core.attributes import AttributeValue


def test_telemetry_config_protocol(mock_telemetry_config) -> None:
    """Test that TelemetryConfig is a Protocol and can be used for type checking."""
    # Since TelemetryConfig is now a Protocol, we can't instantiate it directly
    # Instead, we test that it can be used for type checking with a concrete implementation
    # This should work without type errors
    config: TelemetryConfig = mock_telemetry_config

    assert config.app_type == ApplicationType.TRAINING
    assert config.is_train_iterations_enabled is True
    assert config.is_validation_iterations_enabled is True
    assert config.is_test_iterations_enabled is True
    assert config.is_save_checkpoint_enabled is True
    assert config.custom_metadata is None


def test_telemetry_config_protocol_with_callables(mock_telemetry_config_with_callables) -> None:
    """Test TelemetryConfig protocol with callable-like behavior."""
    config: TelemetryConfig = mock_telemetry_config_with_callables

    assert config.app_type == ApplicationType.VALIDATION
    assert config.is_train_iterations_enabled is False


def test_telemetry_config_protocol_with_metadata(mock_telemetry_config_with_metadata) -> None:
    """Test TelemetryConfig protocol with telemetry metadata."""
    custom_metadata: Dict[str, AttributeValue] = {"telemetry_key": "telemetry_value"}

    config: TelemetryConfig = mock_telemetry_config_with_metadata

    assert config.app_type == ApplicationType.TRAINING
    assert config.custom_metadata == custom_metadata


def test_telemetry_config_runtime_checkable(mock_telemetry_config) -> None:
    """Test that TelemetryConfig is runtime checkable."""
    # Test that isinstance works with the protocol
    assert isinstance(mock_telemetry_config, TelemetryConfig)


def test_telemetry_config_runtime_checkable_with_callables(mock_telemetry_config_with_callables) -> None:
    """Test that TelemetryConfig is runtime checkable with callable implementations."""
    # Test that isinstance works with the protocol
    assert isinstance(mock_telemetry_config_with_callables, TelemetryConfig)


def test_telemetry_config_runtime_checkable_with_metadata(mock_telemetry_config_with_metadata) -> None:
    """Test that TelemetryConfig is runtime checkable with metadata."""
    # Test that isinstance works with the protocol
    assert isinstance(mock_telemetry_config_with_metadata, TelemetryConfig)
