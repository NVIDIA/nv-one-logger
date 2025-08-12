# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OneLogger configuration classes."""

from typing import Dict

import pytest
from pydantic_core._pydantic_core import ValidationError

from nv_one_logger.api.config import LoggerConfig, OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.api.telemetry_config import ApplicationType
from nv_one_logger.core.attributes import AttributeValue


def test_logger_config_invalid_log_level() -> None:
    """Test LoggerConfig validation for invalid log level."""
    with pytest.raises(ValidationError):
        LoggerConfig(log_level="INVALID")  # type: ignore


def test_one_logger_config_default_values() -> None:
    """Test OneLoggerConfig default values."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
    )

    assert config.application_name == "test_app"
    assert config.session_tag == "test_task"
    assert not config.is_baseline_run
    assert config.custom_metadata is None
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
    assert config.enable_for_current_rank
    assert isinstance(config.logger_config, LoggerConfig)
    assert config.summary_data_schema_version == "1.0.0"
    assert config.telemetry_config is None


def test_one_logger_config_with_callables() -> None:
    """Test OneLoggerConfig with callable values."""

    def get_session_tag() -> str:
        return "dynamic_task"

    def get_is_baseline() -> bool:
        return True

    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn=get_session_tag,
        is_baseline_run_or_fn=get_is_baseline,
        world_size_or_fn=4,
    )

    assert config.session_tag == "dynamic_task"
    assert config.is_baseline_run is True


def test_one_logger_config_with_custom_metadata() -> None:
    """Test OneLoggerConfig with custom metadata."""
    custom_metadata: Dict[str, AttributeValue] = {"key1": "value1", "key2": "value2"}

    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        custom_metadata=custom_metadata,
        world_size_or_fn=4,
    )

    assert config.custom_metadata == custom_metadata


def test_one_logger_config_with_telemetry_config(mock_telemetry_config) -> None:
    """Test OneLoggerConfig with telemetry_config."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        telemetry_config=mock_telemetry_config,
        world_size_or_fn=4,
    )

    assert config.telemetry_config is not None
    assert config.telemetry_config.app_type == ApplicationType.TRAINING


def test_one_logger_config_model_validator_empty_application_name() -> None:
    """Test that OneLoggerConfig raises ValidationError when application_name is empty."""
    with pytest.raises(ValidationError, match="application_name cannot be empty or whitespace-only"):
        OneLoggerConfig(
            application_name="",
            session_tag_or_fn="test_task",
            world_size_or_fn=4,
        )


def test_one_logger_config_model_validator_whitespace_only_application_name() -> None:
    """Test that OneLoggerConfig raises ValidationError when application_name is only whitespace."""
    with pytest.raises(ValidationError, match="application_name cannot be empty or whitespace-only"):
        OneLoggerConfig(
            application_name="   \t\n  ",
            session_tag_or_fn="test_task",
            world_size_or_fn=4,
        )


def test_one_logger_config_model_validator_empty_custom_metadata_key() -> None:
    """Test that OneLoggerConfig raises ValidationError when custom_metadata has empty keys."""
    custom_metadata: Dict[str, AttributeValue] = {"": "value1", "key2": "value2"}

    with pytest.raises(ValidationError, match="custom_metadata keys must be non-empty strings"):
        OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_task",
            custom_metadata=custom_metadata,
            world_size_or_fn=4,
        )


def test_one_logger_config_model_validator_whitespace_only_custom_metadata_key() -> None:
    """Test that OneLoggerConfig raises ValidationError when custom_metadata has whitespace-only keys."""
    custom_metadata: Dict[str, AttributeValue] = {"   ": "value1", "key2": "value2"}

    with pytest.raises(ValidationError, match="custom_metadata keys must be non-empty strings"):
        OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_task",
            custom_metadata=custom_metadata,
            world_size_or_fn=4,
        )


def test_one_logger_config_model_validator_valid_custom_metadata() -> None:
    """Test that OneLoggerConfig accepts valid custom_metadata."""
    custom_metadata: Dict[str, AttributeValue] = {"key1": "value1", "key2": "value2", "key_with_spaces": "value3"}

    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        custom_metadata=custom_metadata,
        world_size_or_fn=4,
    )

    assert config.custom_metadata == custom_metadata
