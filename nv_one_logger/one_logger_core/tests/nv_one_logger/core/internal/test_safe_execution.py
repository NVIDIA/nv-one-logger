# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the safe execution utilities."""

from typing import Generator, Tuple, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from nv_one_logger.api.config import OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.internal.safe_execution import exception_guard, safely_execute


@pytest.fixture(autouse=True)
def mock_provider() -> Generator[None, None, None]:
    """Mock OneLoggerProvider.instance() to return a mock provider.

    This fixture is automatically used for all tests.
    """
    mock = MagicMock(spec=OneLoggerProvider)
    mock.config = MagicMock(spec=OneLoggerConfig)
    mock.config.error_handling_strategy = OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
    mock.config.enable_one_logger = True
    mock.force_disable_logging = MagicMock()  # Explicitly create the mock method

    with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock):
        yield


@exception_guard
def _guarded_normal_function() -> str:
    return "success"


@exception_guard
def _guarded_failing_function() -> str:
    raise ValueError("Test error")


class TestExceptionGuard:
    """Tests for the exception_guard decorator."""

    def test_propagate_exceptions_strategy(self, mock_provider: Mock) -> None:
        """Test that exceptions are propagated when using PROPAGATE_EXCEPTIONS strategy."""
        provider = cast(MagicMock, OneLoggerProvider.instance())
        provider.config.error_handling_strategy = OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
        provider.config.enable_one_logger = True

        with pytest.raises(ValueError, match="Test error"):
            _guarded_failing_function()
        provider.force_disable_logging.assert_not_called()

    def test_disable_quietly_and_report_metric_corruption_strategy(self) -> None:
        """Test that exceptions are caught and OneLogger is disabled when using DISABLE_QUIETLY_AND_REPORT_METRIC_CORRUPTION strategy."""
        provider = cast(MagicMock, OneLoggerProvider.instance())
        provider.config.error_handling_strategy = OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
        provider.config.enable_one_logger = True

        result = _guarded_failing_function()
        assert result is None
        provider.force_disable_logging.assert_called_once()

    @pytest.mark.parametrize(
        "strategy",
        [
            OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS,
            OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,
        ],
    )
    def test_successful_execution(self, strategy: OneLoggerErrorHandlingStrategy) -> None:
        """Test that the function executes successfully when no exception occurs."""
        provider = cast(MagicMock, OneLoggerProvider.instance())
        provider.config.error_handling_strategy = strategy
        provider.config.enable_one_logger = True

        result = _guarded_normal_function()
        assert result == "success"
        provider.force_disable_logging.assert_not_called()


@safely_execute
def _safely_executed_normal_function() -> str:
    return "success"


@safely_execute
def _safely_executed_failing_function() -> str:
    raise ValueError("Test error")


class TestSafelyExecute:
    """Tests for the safely_execute decorator."""

    def test_safely_execute_when_enabled(self) -> None:
        """Test that the decorated function executes when OneLogger is enabled."""
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = True
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):

            # Execute
            result = _safely_executed_normal_function()

            # Verify
            assert result == "success"

    def test_safely_execute_when_disabled(self) -> None:
        """Test that the decorated function returns None when OneLogger is disabled."""
        # Setup
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = False
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):

            # Execute
            result = _safely_executed_normal_function()

            # Verify
            assert result is None

    def test_safely_execute_with_args(self) -> None:
        """Test that the decorated function executes with arguments when OneLogger is enabled."""
        # Setup
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = True
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):

            @safely_execute
            def test_func(arg1: str, arg2: int) -> Tuple[str, int]:
                return arg1, arg2

            # Execute
            result = test_func("test", 42)

            # Verify
            assert result == ("test", 42)

    def test_safely_execute_with_kwargs(self) -> None:
        """Test that the decorated function executes with keyword arguments when OneLogger is enabled."""
        # Setup
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = True
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):

            @safely_execute
            def test_func(arg1: str, arg2: int) -> Tuple[str, int]:
                return arg1, arg2

            # Execute
            result = test_func(arg1="test", arg2=42)

            # Verify
            assert result == ("test", 42)

    def test_safely_execute_with_args_and_kwargs(self) -> None:
        """Test that the decorated function executes with both args and kwargs when OneLogger is enabled."""
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = True
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):

            @safely_execute
            def test_func(arg1: str, arg2: int, kwarg1: str = "default") -> Tuple[str, int, str]:
                return arg1, arg2, kwarg1

            # Execute
            result = test_func("test", 42, kwarg1="custom")

            # Verify
            assert result == ("test", 42, "custom")

    def test_safely_execute_with_args_and_kwargs_when_disabled(self) -> None:
        """Test that the decorated function returns None with both args and kwargs when OneLogger is disabled."""
        # Setup
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = False
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):
            # Create a test function
            @safely_execute
            def test_func(arg1: str, arg2: int, kwarg1: str = "default") -> Tuple[str, int, str]:
                return arg1, arg2, kwarg1

            # Execute
            result = test_func("test", 42, kwarg1="custom")

            # Verify
            assert result is None

    def test_safely_execute_with_return_type(self) -> None:
        """Test that the decorated function preserves its return type annotation."""
        # Setup
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = True
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):
            # Verify
            assert _safely_executed_normal_function.__annotations__["return"] == str
            assert _safely_executed_normal_function() == "success"

    def test_safely_execute_with_return_type_when_disabled(self) -> None:
        """Test that the decorated function preserves its return type annotation even when disabled."""
        # Setup
        mock_provider = MagicMock(spec=OneLoggerProvider)
        mock_provider.one_logger_enabled = False
        with patch("nv_one_logger.core.internal.safe_execution.OneLoggerProvider.instance", return_value=mock_provider):

            # Execute
            result = _safely_executed_normal_function()

            # Verify
            assert _safely_executed_normal_function.__annotations__["return"] == str
            assert result is None

    def test_propagate_exceptions_strategy(self) -> None:
        """Test that exceptions are propagated when using PROPAGATE_EXCEPTIONS strategy."""
        provider = cast(MagicMock, OneLoggerProvider.instance())
        provider.config.error_handling_strategy = OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
        provider.config.enable_one_logger = True

        with pytest.raises(ValueError, match="Test error"):
            _safely_executed_failing_function()
        provider.force_disable_logging.assert_not_called()

    def test_disable_quietly_and_report_metric_corruption_strategy(self) -> None:
        """Test that exceptions are caught and OneLogger is disabled when using DISABLE_QUIETLY_AND_REPORT_METRIC_CORRUPTION strategy."""
        provider = cast(MagicMock, OneLoggerProvider.instance())
        provider.config.error_handling_strategy = OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR
        provider.config.enable_one_logger = True

        result = _safely_executed_failing_function()
        assert result is None
        provider.force_disable_logging.assert_called_once()

    @pytest.mark.parametrize(
        "strategy",
        [
            OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS,
            OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,
        ],
    )
    def test_successful_execution(self, strategy: OneLoggerErrorHandlingStrategy) -> None:
        """Test that the function executes successfully when no exception occurs."""
        provider = cast(MagicMock, OneLoggerProvider.instance())
        provider.config.error_handling_strategy = strategy
        provider.config.enable_one_logger = True

        result = _safely_executed_normal_function()
        assert result == "success"
        provider.force_disable_logging.assert_not_called()
