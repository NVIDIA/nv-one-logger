# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OneLogger configuration classes."""

from typing import Dict

import pytest

from nv_one_logger.api.config import ApplicationType, LoggerConfig, OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.core.attributes import AttributeValue


def test_logger_config_invalid_log_level() -> None:
    """Test LoggerConfig validation for invalid log level."""
    with pytest.raises(ValueError, match="log_level must be one of"):
        LoggerConfig(log_level="INVALID")  # type: ignore


def test_one_logger_config_default_values() -> None:
    """Test OneLoggerConfig default values."""
    config = OneLoggerConfig(
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_task",
        app_type_or_fn=ApplicationType.TRAINING,
    )

    assert config.application_name == "test_app"
    assert config.perf_tag == "test_perf"
    assert config.session_tag == "test_task"
    assert config.app_type == ApplicationType.TRAINING
    assert not config.is_baseline_run
    assert config.custom_metadata is None
    assert config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
    assert config.enable_one_logger
    assert isinstance(config.logger_config, LoggerConfig)


def test_one_logger_config_with_callables() -> None:
    """Test OneLoggerConfig with callable values."""

    def get_perf_tag() -> str:
        return "dynamic_version2_batchsize128"

    def get_session_tag() -> str:
        return "dynamic_task"

    def get_app_type() -> ApplicationType:
        return ApplicationType.VALIDATION

    def get_is_baseline() -> bool:
        return True

    config = OneLoggerConfig(
        application_name="test_app",
        perf_tag_or_fn=get_perf_tag,
        session_tag_or_fn=get_session_tag,
        app_type_or_fn=get_app_type,
        is_baseline_run_or_fn=get_is_baseline,
    )

    assert config.perf_tag == "dynamic_version2_batchsize128"
    assert config.session_tag == "dynamic_task"
    assert config.app_type == ApplicationType.VALIDATION
    assert config.is_baseline_run is True


def test_one_logger_config_with_custom_metadata() -> None:
    """Test OneLoggerConfig with custom metadata."""
    custom_metadata: Dict[str, AttributeValue] = {"key1": "value1", "key2": "value2"}

    config = OneLoggerConfig(
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_task",
        app_type_or_fn=ApplicationType.TRAINING,
        custom_metadata=custom_metadata,
    )

    assert config.custom_metadata == custom_metadata
