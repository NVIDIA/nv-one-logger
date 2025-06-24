# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the utils module."""

import os

import pytest

from nv_one_logger.core.internal.utils import evaluate_value, temporarily_modify_env


class TestTemporaryModifyEnvVar:
    """Test class for temporarily_modify_env tests."""

    @pytest.fixture
    def env_var(self) -> str:
        """Fixture that provides a test environment variable name."""
        return "TEST_VAR"

    def test_set_new_value(self, env_var: str) -> None:
        """Test setting a new environment variable value."""
        new_value = "new_value"

        with temporarily_modify_env(env_var, new_value):
            assert os.environ[env_var] == new_value

        assert env_var not in os.environ

    def test_remove_existing_value(self, env_var: str) -> None:
        """Test removing an existing environment variable."""
        original_value = "original_value"
        os.environ[env_var] = original_value

        with temporarily_modify_env(env_var, None):
            assert env_var not in os.environ

        assert os.environ[env_var] == original_value

    def test_restore_original_value(self, env_var: str) -> None:
        """Test that original value is restored after context manager exit."""
        original_value = "original_value"
        new_value = "new_value"
        os.environ[env_var] = original_value

        with temporarily_modify_env(env_var, new_value):
            assert os.environ[env_var] == new_value

        assert os.environ[env_var] == original_value


class TestEvaluateValue:
    """Test class for evaluate_value function tests."""

    def test_direct_value(self) -> None:
        """Test that a direct value is returned unchanged."""
        value = 42
        result = evaluate_value(value)
        assert result == 42

    def test_callable_value(self) -> None:
        """Test that a callable is evaluated and its result is returned."""

        def get_value() -> int:
            return 42

        result = evaluate_value(get_value)
        assert result == 42

    def test_none_value(self) -> None:
        """Test that None is handled correctly."""
        result = evaluate_value(None)
        assert result is None

    def test_callable_returning_none(self) -> None:
        """Test that a callable returning None is handled correctly."""

        def return_none() -> None:
            return None

        result = evaluate_value(return_none)
        assert result is None
