# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the utils module."""

import os
from typing import Any, Callable

import pytest

from nv_one_logger.core.internal.utils import evaluate_value, patch_function, patch_method, temporarily_modify_env


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


class TestPatchFunction:
    """Test class for patch_function tests."""

    def test_patch_function_preserves_function_metadata(self) -> None:
        """Test that patch_function correctly wraps a function with a simple wrapper while preserving the original function's metadata."""

        def original_func(a: int, b: int) -> int:
            """Original function docstring."""
            return a + b

        def wrapper_func(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            return original(*args, **kwargs) * 2

        patched_func = patch_function(original_func)(wrapper_func)

        assert patched_func(3, 5) == 16  # (3 + 5) * 2
        assert patched_func.__name__ == "original_func"
        assert patched_func.__doc__ == "Original function docstring."

    def test_patch_function_with_logging_wrapper(self) -> None:
        """Test that patch_function works with a logging wrapper."""

        def original_func(x: int) -> int:
            return x * 2

        log_calls: list[str] = []

        def logging_wrapper(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            log_calls.append(f"Calling with args: {args}, kwargs: {kwargs}")
            result = original(*args, **kwargs)
            log_calls.append(f"Result: {result}")
            return result

        patched_func = patch_function(original_func)(logging_wrapper)

        result = patched_func(10)

        assert result == 20
        assert len(log_calls) == 2
        assert log_calls[0] == "Calling with args: (10,), kwargs: {}"
        assert log_calls[1] == "Result: 20"

    def test_patch_function_with_kwargs(self) -> None:
        """Test that patch_function correctly handles keyword arguments."""

        def original_func(a: int, b: int, multiplier: int = 1) -> int:
            return (a + b) * multiplier

        def wrapper_func(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            return original(*args, **kwargs) + 100

        patched_func = patch_function(original_func)(wrapper_func)

        result = patched_func(3, 5, multiplier=2)
        assert result == 116  # ((3 + 5) * 2) + 100

    def test_patch_function_with_exception_handling(self) -> None:
        """Test that patch_function works with exception handling in wrapper."""

        def original_func(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 2

        def exception_wrapper(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            try:
                return original(*args, **kwargs)
            except ValueError as e:
                return f"Caught error: {e}"

        patched_func = patch_function(original_func)(exception_wrapper)

        # Test normal case
        result = patched_func(5)
        assert result == 10

        # Test exception case
        result = patched_func(-5)
        assert result == "Caught error: Negative value not allowed"


class TestPatchMethod:
    """Test class for patch_method tests."""

    def test_patch_method_with_simple_wrapper(self) -> None:
        """Test that patch_method correctly wraps a method with a simple wrapper while preserving the original method's metadata."""

        class TestClass:
            def __init__(self, value: int) -> None:
                self.value = value

            def original_func(self, x: int) -> int:
                """Original function docstring."""
                return self.value + x

        def wrapper_func(original: Callable[..., Any], self: Any, *args: Any, **kwargs: Any) -> Any:
            return original(self, *args, **kwargs) * 2

        TestClass.original_func = patch_method(TestClass.original_func)(wrapper_func)

        obj = TestClass(10)
        result = obj.original_func(5)
        assert result == 30  # (10 + 5) * 2
        assert TestClass.original_func.__name__ == "original_func"
        assert TestClass.original_func.__doc__ == "Original function docstring."

    def test_patch_method_with_logging_wrapper(self) -> None:
        """Test that patch_method works with a logging wrapper."""

        class TestClass:
            def __init__(self, name: str) -> None:
                self.name = name

            def greet(self, greeting: str) -> str:
                return f"{greeting}, {self.name}!"

        log_calls: list[str] = []

        def logging_wrapper(original: Callable[..., Any], self: Any, *args: Any, **kwargs: Any) -> Any:
            log_calls.append(f"Calling {original.__name__} on {self.name} with args: {args}, kwargs: {kwargs}")
            result = original(self, *args, **kwargs)
            log_calls.append(f"Result: {result}")
            return result

        TestClass.greet = patch_method(TestClass.greet)(logging_wrapper)

        obj = TestClass("Alice")
        result = obj.greet("Hello")

        assert result == "Hello, Alice!"
        assert len(log_calls) == 2
        assert log_calls[0] == "Calling greet on Alice with args: ('Hello',), kwargs: {}"
        assert log_calls[1] == "Result: Hello, Alice!"

    def test_patch_method_with_kwargs(self) -> None:
        """Test that patch_method correctly handles keyword arguments."""

        class TestClass:
            def calculate(self, a: int, b: int, multiplier: int = 1) -> int:
                return (a + b) * multiplier

        def wrapper_func(original: Callable[..., Any], self: Any, *args: Any, **kwargs: Any) -> Any:
            return original(self, *args, **kwargs) + 100

        TestClass.calculate = patch_method(TestClass.calculate)(wrapper_func)

        obj = TestClass()
        result = obj.calculate(3, 5, multiplier=2)
        assert result == 116  # ((3 + 5) * 2) + 100

    def test_patch_method_with_exception_handling(self) -> None:
        """Test that patch_method works with exception handling in wrapper."""

        class TestClass:
            def divide(self, x: int, y: int) -> float:
                if y == 0:
                    raise ZeroDivisionError("Division by zero")
                return x / y

        def exception_wrapper(original: Callable[..., Any], self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original(self, *args, **kwargs)
            except ZeroDivisionError as e:
                return f"Caught error: {e}"

        TestClass.divide = patch_method(TestClass.divide)(exception_wrapper)

        obj = TestClass()

        # Test normal case
        result = obj.divide(10, 2)
        assert result == 5.0

        # Test exception case
        result = obj.divide(10, 0)
        assert result == "Caught error: Division by zero"

    def test_patch_method_with_multiple_instances(self) -> None:
        """Test that patch_method works correctly with multiple class instances."""

        class TestClass:
            def __init__(self, value: int) -> None:
                self.value = value

            def get_value(self) -> int:
                return self.value

        def wrapper_func(original: Callable[..., Any], self: Any, *args: Any, **kwargs: Any) -> Any:
            return original(self, *args, **kwargs) + 100

        TestClass.get_value = patch_method(TestClass.get_value)(wrapper_func)

        obj1 = TestClass(10)
        obj2 = TestClass(20)

        assert obj1.get_value() == 110  # 10 + 100
        assert obj2.get_value() == 120  # 20 + 100
