# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OneLogger internal logging functionality."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nv_one_logger.api.config import LoggerConfig, OneLoggerConfig
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.internal.logging import get_logger


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_one_logger_provider(temp_log_dir):
    """Create a mock OneLoggerProvider with proper configuration."""
    # Reset the singleton to ensure clean state
    if hasattr(OneLoggerProvider, "_instances"):
        OneLoggerProvider._instances.clear()

    # Create config with temporary log paths
    config = OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        world_size_or_fn=1,
        logger_config=LoggerConfig(
            log_file_path_for_info=str(temp_log_dir / "info.log"),
            log_file_path_for_err=str(temp_log_dir / "error.log"),
            log_level="INFO",
        ),
    )

    # Configure OneLoggerProvider
    mock_recorder = MagicMock()
    provider = OneLoggerProvider.instance()
    provider.configure(config, mock_recorder)

    yield provider

    # Clean up
    if hasattr(OneLoggerProvider, "_instances"):
        OneLoggerProvider._instances.clear()


def test_get_logger_creates_handlers_on_first_call(mock_one_logger_provider):
    """Test that get_logger creates file handlers on the first call."""
    logger_name = "test.module.first_call"

    # Clear any existing handlers for this logger name
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()

    # First call should create handlers
    logger = get_logger(logger_name)

    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2  # Should have info and error handlers

    # Verify handler types and levels
    handler_levels = [handler.level for handler in logger.handlers]
    assert logging.INFO in handler_levels
    assert logging.ERROR in handler_levels

    # Verify handlers are FileHandlers
    for handler in logger.handlers:
        assert isinstance(handler, logging.FileHandler)


def test_get_logger_does_not_duplicate_handlers(mock_one_logger_provider):
    """Test that calling get_logger multiple times does NOT create duplicate handlers."""
    logger_name = "test.module.no_duplicates"

    # Clear any existing handlers for this logger name
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()

    # First call should create handlers
    logger1 = get_logger(logger_name)
    initial_handler_count = len(logger1.handlers)
    assert initial_handler_count == 2  # info and error handlers

    # Store references to the original handlers
    original_handlers = logger1.handlers.copy()

    # Second call should NOT create new handlers
    logger2 = get_logger(logger_name)
    assert logger2 is logger1  # Should be the same logger instance
    assert len(logger2.handlers) == initial_handler_count

    # Verify the handlers are the exact same objects
    assert logger2.handlers == original_handlers

    # Third call should also not create new handlers
    logger3 = get_logger(logger_name)
    assert logger3 is logger1
    assert len(logger3.handlers) == initial_handler_count
    assert logger3.handlers == original_handlers


def test_get_logger_multiple_calls_same_handler_references(mock_one_logger_provider):
    """Test that multiple calls to get_logger return the same handler references."""
    logger_name = "test.module.same_references"

    # Clear any existing handlers
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()

    # Get logger multiple times
    logger1 = get_logger(logger_name)
    logger2 = get_logger(logger_name)
    logger3 = get_logger(logger_name)

    # All should be the same logger
    assert logger1 is logger2 is logger3

    # All should have the same number of handlers
    assert len(logger1.handlers) == len(logger2.handlers) == len(logger3.handlers) == 2

    # Verify the handlers are the exact same objects (not copies)
    for i, handler in enumerate(logger1.handlers):
        assert logger2.handlers[i] is handler
        assert logger3.handlers[i] is handler


def test_get_logger_different_loggers_get_own_handlers(mock_one_logger_provider):
    """Test that different logger names get their own set of handlers."""
    logger_name1 = "test.module.logger1"
    logger_name2 = "test.module.logger2"

    # Clear any existing handlers
    logging.getLogger(logger_name1).handlers.clear()
    logging.getLogger(logger_name2).handlers.clear()

    # Get two different loggers
    logger1 = get_logger(logger_name1)
    logger2 = get_logger(logger_name2)

    # They should be different logger instances
    assert logger1 is not logger2

    # Both should have handlers
    assert len(logger1.handlers) == 2
    assert len(logger2.handlers) == 2

    # Handlers should be different objects
    for handler1 in logger1.handlers:
        for handler2 in logger2.handlers:
            assert handler1 is not handler2


def test_get_logger_without_one_logger_ready():
    """Test that get_logger returns a logger without handlers when OneLogger is not ready."""
    # Reset the singleton
    if hasattr(OneLoggerProvider, "_instances"):
        OneLoggerProvider._instances.clear()

    logger_name = "test.module.not_ready"

    # Clear any existing handlers
    logging.getLogger(logger_name).handlers.clear()

    # Get logger without configuring OneLoggerProvider
    logger = get_logger(logger_name)

    assert logger is not None
    assert isinstance(logger, logging.Logger)
    # Should not have added handlers since OneLogger is not ready
    assert len(logger.handlers) == 0


def test_get_logger_handlers_have_correct_configuration(mock_one_logger_provider):
    """Test that handlers are configured with the correct log levels and formatters."""
    logger_name = "test.module.config_check"

    # Clear any existing handlers
    logging.getLogger(logger_name).handlers.clear()

    # Get logger
    logger = get_logger(logger_name)

    # Verify we have exactly 2 handlers
    assert len(logger.handlers) == 2

    # Find the info and error handlers
    info_handler = None
    error_handler = None

    for handler in logger.handlers:
        if handler.level == logging.INFO:
            info_handler = handler
        elif handler.level == logging.ERROR:
            error_handler = handler

    # Verify both handlers exist
    assert info_handler is not None, "INFO handler should exist"
    assert error_handler is not None, "ERROR handler should exist"

    # Verify formatters are set
    assert info_handler.formatter is not None
    assert error_handler.formatter is not None

    # Verify both handlers are FileHandlers
    assert isinstance(info_handler, logging.FileHandler)
    assert isinstance(error_handler, logging.FileHandler)


def test_get_logger_handlers_not_recreated_after_manual_clear(mock_one_logger_provider):
    """Test that handlers are recreated if they are manually cleared."""
    logger_name = "test.module.manual_clear"

    # Clear any existing handlers
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()

    # First call creates handlers
    logger1 = get_logger(logger_name)
    assert len(logger1.handlers) == 2

    # Manually clear handlers (simulating external modification)
    logger1.handlers.clear()
    assert len(logger1.handlers) == 0

    # Next call should recreate handlers since the list is now empty
    logger2 = get_logger(logger_name)
    assert logger2 is logger1  # Same logger instance
    assert len(logger2.handlers) == 2  # Handlers recreated


def test_get_logger_prevents_file_handle_leak(mock_one_logger_provider):
    """Test that multiple calls don't create file handle leaks."""
    logger_name = "test.module.no_leak"

    # Clear any existing handlers
    logging.getLogger(logger_name).handlers.clear()

    # Call get_logger multiple times
    for _ in range(10):
        logger = get_logger(logger_name)

    # Should still only have 2 handlers, not 20
    assert len(logger.handlers) == 2

    # Verify handlers are FileHandlers (file handles)
    file_handler_count = sum(1 for h in logger.handlers if isinstance(h, logging.FileHandler))
    assert file_handler_count == 2
