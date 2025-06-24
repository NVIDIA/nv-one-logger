# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OneLoggerProvider class."""

import pytest

from nv_one_logger.api.one_logger_provider import OneLoggerProvider


def test_singleton_instance() -> None:
    """Test that the provider maintains singleton behavior."""
    # Get a new instance
    assert OneLoggerProvider.instance() == OneLoggerProvider.instance()

    # No direct instantiation allowed
    with pytest.raises(TypeError, match="OneLoggerProvider cannot be instantiated directly."):
        _ = OneLoggerProvider()
