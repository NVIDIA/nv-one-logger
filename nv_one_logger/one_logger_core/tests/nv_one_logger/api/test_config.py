# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OneLogger configuration classes."""

import os
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from pydantic_core._pydantic_core import ValidationError

from nv_one_logger.api.config import LoggerConfig, OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.api.telemetry_config import ApplicationType
from nv_one_logger.core.attributes import AttributeValue
from nv_one_logger.core.internal.logging import get_logger


def test_logger_config_invalid_log_level() -> None:
    """Test LoggerConfig validation for invalid log level."""
    with pytest.raises(ValidationError):
        LoggerConfig(log_level="INVALID")  # type: ignore


def test_one_logger_config_default_values() -> None:
    """Test OneLoggerConfig default values for enabled rank."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        # enable_for_current_rank defaults to True
    )

    assert config.application_name == "test_app"
    assert config.session_tag == "test_task"
    assert not config.is_baseline_run
    assert config.custom_metadata is None
    # Enabled rank should keep default PROPAGATE_EXCEPTIONS
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


def test_one_logger_config_enabled_rank_default_error_handling() -> None:
    """Test that enabled ranks keep default PROPAGATE_EXCEPTIONS when not explicitly set."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=True,
        # No explicit error_handling_strategy provided
    )

    # Enabled ranks should keep the default PROPAGATE_EXCEPTIONS
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
    assert config.enable_for_current_rank is True


def test_one_logger_config_disabled_rank_auto_quiet_defaults() -> None:
    """Test that disabled ranks automatically get quiet error handling when not explicitly set."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
        # No explicit error_handling_strategy provided
    )

    # Disabled ranks should auto-apply quiet error handling for safety
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
    assert config.enable_for_current_rank is False


def test_one_logger_config_disabled_rank_user_override_propagate() -> None:
    """Test that disabled ranks respect user override to PROPAGATE_EXCEPTIONS."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
        error_handling_strategy=OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS,  # Explicit override
    )

    # User override should be respected even for disabled ranks
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
    assert config.enable_for_current_rank is False


def test_one_logger_config_disabled_rank_user_override_quiet() -> None:
    """Test that disabled ranks respect user override to DISABLE_QUIETLY (explicit same as default)."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
        error_handling_strategy=OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,  # Explicit choice
    )

    # Explicit choice should be respected (no auto-application message)
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
    assert config.enable_for_current_rank is False


def test_one_logger_config_enabled_rank_user_override_quiet() -> None:
    """Test that enabled ranks respect user override to DISABLE_QUIETLY."""
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=True,
        error_handling_strategy=OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,  # Explicit override
    )

    # User override should be respected for enabled ranks too
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
    assert config.enable_for_current_rank is True


def test_one_logger_config_user_intent_detection() -> None:
    """Test that the config properly detects when user explicitly sets error handling strategy."""
    # Test with explicit strategy
    config_explicit = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
        error_handling_strategy=OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS,
    )

    # Should detect user intent and respect the explicit choice
    assert config_explicit.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS

    # Test without explicit strategy
    config_implicit = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
        # No error_handling_strategy provided
    )

    # Should apply auto defaults for disabled rank
    assert config_implicit.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR


def test_one_logger_config_defaults_backward_compatibility() -> None:
    """Test that existing code patterns continue to work unchanged."""
    # This represents typical existing code that should work exactly the same
    config = OneLoggerConfig(
        application_name="legacy_app",
        session_tag_or_fn="legacy_session",
        world_size_or_fn=1,
        enable_for_current_rank=True,
        error_handling_strategy=OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,
    )

    # Should work exactly as before - no changes to existing behavior
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
    assert config.enable_for_current_rank is True
    assert config.application_name == "legacy_app"
    assert config.session_tag == "legacy_session"


@patch.dict(os.environ, {"RANK": "5"})
@patch("nv_one_logger.api.config.logger")
def test_disabled_rank_logs_info_and_sets_critical_level(mock_logger) -> None:
    """Test that disabling OneLogger for a rank logs INFO message and sets log level to CRITICAL.

    This test verifies:
    1. An INFO message is logged when OneLogger is disabled for a rank (with rank number)
    2. The log level is automatically set to CRITICAL to suppress further OneLogger logs
    3. The error handling strategy is set to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
    """
    # Create config with disabled rank and no explicit error_handling_strategy or logger_config
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
    )

    # Verify the WARNING message was logged with rank information
    mock_logger.warning.assert_called_once()
    warning_call_args = mock_logger.warning.call_args[0][0]
    assert "Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR" in warning_call_args
    assert "rank=5" in warning_call_args
    assert "OneLogger disabled" in warning_call_args

    # Verify error handling strategy was set
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR

    # Verify log level was automatically set to CRITICAL to suppress further logs
    assert config.logger_config.log_level == "CRITICAL"

    # Verify rank is disabled
    assert config.enable_for_current_rank is False


@patch.dict(os.environ, {"RANK": "3"})
@patch("nv_one_logger.api.config.logger")
def test_disabled_rank_respects_user_log_level(mock_logger) -> None:
    """Test that user-provided log level is respected even for disabled ranks."""
    # Create config with disabled rank but explicit logger_config
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
        logger_config=LoggerConfig(log_level="DEBUG"),  # User explicitly wants DEBUG
    )

    # Verify the WARNING message was still logged
    mock_logger.warning.assert_called_once()

    # Verify error handling strategy was set
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR

    # Verify user's log level was respected (not overridden to CRITICAL)
    assert config.logger_config.log_level == "DEBUG"

    # Verify rank is disabled
    assert config.enable_for_current_rank is False


@patch("nv_one_logger.api.config.logger")
def test_enabled_rank_does_not_log_or_change_log_level(mock_logger) -> None:
    """Test that enabled ranks don't log the INFO message or change log level."""
    # Create config with enabled rank (default)
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=True,
    )

    # Verify NO info message was logged for enabled ranks
    mock_logger.info.assert_not_called()

    # Verify error handling strategy keeps default for enabled ranks
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS

    # Verify log level remains at default INFO (not changed to CRITICAL)
    assert config.logger_config.log_level == "INFO"

    # Verify rank is enabled
    assert config.enable_for_current_rank is True


@patch.dict(os.environ, {"RANK": "7"})
@patch("nv_one_logger.api.config.logger")
def test_disabled_rank_suppresses_further_onelogger_logs(mock_config_logger) -> None:
    """Test that CRITICAL log level actually suppresses OneLogger's internal logs.

    This test demonstrates that when log_level is set to CRITICAL for a disabled rank,
    OneLogger's internal logs at INFO, WARNING, and ERROR levels are suppressed,
    and only CRITICAL logs would go through.
    """
    # Create config with disabled rank - this will set log_level to CRITICAL
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_task",
        world_size_or_fn=4,
        enable_for_current_rank=False,
    )

    # Verify initial WARNING message was logged during config creation
    mock_config_logger.warning.assert_called_once()
    assert "rank=7" in mock_config_logger.warning.call_args[0][0]

    # Verify log level is CRITICAL
    assert config.logger_config.log_level == "CRITICAL"

    # Now simulate OneLogger's internal logging with this config
    # Create a mock recorder to initialize OneLoggerProvider
    mock_recorder = MagicMock()

    # Reset the singleton to ensure clean state
    if hasattr(OneLoggerProvider, "_instances"):
        OneLoggerProvider._instances.clear()

    # Configure OneLoggerProvider with our config
    OneLoggerProvider.instance().configure(config, mock_recorder)

    # Get a logger using OneLogger's get_logger() - this respects the config's log_level
    test_logger = get_logger("test_module")

    # The logger's level should be set to CRITICAL (this is what suppresses logs)
    assert test_logger.level == 50  # CRITICAL = 50

    # Verify that logging at various levels would respect the CRITICAL threshold
    # (We don't need to mock since the main assertion is the log level itself)
    test_logger.debug("This debug message should be suppressed")
    test_logger.info("This info message should be suppressed")
    test_logger.warning("This warning message should be suppressed")
    test_logger.error("This error message should be suppressed")
    test_logger.critical("This critical message should go through")

    # Confirm CRITICAL level is enforced (the key test for log suppression)
    assert test_logger.level == 50

    # Clean up singleton
    if hasattr(OneLoggerProvider, "_instances"):
        OneLoggerProvider._instances.clear()
