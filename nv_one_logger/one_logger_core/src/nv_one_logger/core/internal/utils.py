# SPDX-License-Identifier: Apache-2.0
"""Contains utilities for the core module."""

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar, Union


@contextmanager
def temporarily_modify_env(var_name: str, new_var: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily modify an environment variable. The original value is restored when the context manager is exited.

    Args:
        var_name: The name of the environment variable to modify.
        new_var: The new value to set the environment variable to.
    """
    original_value = os.environ.get(var_name)

    if new_var is not None:
        os.environ[var_name] = new_var
    else:
        os.environ.pop(var_name, None)

    try:
        yield
    finally:
        if original_value is not None:
            os.environ[var_name] = original_value
        elif new_var is not None:
            os.environ.pop(var_name, None)


_T = TypeVar("_T")


def evaluate_value(value: Union[_T, Callable[[], _T]]) -> _T:
    """Evaluate a value that could be either a direct value or a callable.

    Args:
        value: Either a direct value or a callable that returns a value.

    Returns:
        The evaluated value.
    """
    if callable(value):
        return value()
    return value


def patch_function(original_func: Callable[..., Any]) -> Callable[..., Any]:
    """Patches a given function by allowing a caller to provide a custom wrapper.

    Args:
        original_func: The original function to be patched.

    Returns:
        A new function with the patched behavior.


    Example usage:
    # original function
    def my_func(a, b):
        return a + b

    # Define a wrapper that adds logging and calls the original function
    def my_wrapper(original_func, *args, **kwargs):
        print(f"Calling {original_func.__name__} with arguments: {args}, {kwargs}")
        result = original_func(*args, **kwargs)  # Call the original function
        print(f"Result: {result}")
        return result

    # Patch the original function using the utility
    my_func = patch_function(my_func)(my_wrapper)

    # Calling my_func will now runs the extra logging logic in my_wrapper before calling the original function and prints the result after.
    result = my_func(3, 5)
    """

    def patch(wrapper_func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original_func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Call the wrapper provided by the caller
            return wrapper_func(original_func, *args, **kwargs)

        return wrapped

    return patch


def patch_method(original_method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Patches a given object method by allowing a caller to provide a custom wrapper.

    Args:
        original_method: The original method to be patched.

    Returns:
        A new method with the patched behavior.

    # Usage Example:

    class MyClass:
        def __init__(self, name):
            self.name = name

        def greet(self, greeting):
            return f"{greeting}, {self.name}!"

    # Define a wrapper that adds logging and calls the original method
    def greet_wrapper(original_method: Callable, self: Any, *args: Any, **kwargs: Any) -> Any:
        print(f"Calling {original_method.__name__} on {self} with arguments: {args}, {kwargs}")
        result = original_method(self, *args, **kwargs)  # Call the original method with "self"
        print(f"Result: {result}")
        return result

    # Patch the greet method of MyClass using the utility
    MyClass.greet = patch_method(MyClass.greet)(greet_wrapper)

    # Test the patched method
    obj = MyClass("Alice")
    result = obj.greet("Hello")
    """

    def patch(wrapper_func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original_method)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Call the wrapper provided by the caller with "self" as the first argument
            return wrapper_func(original_method, self, *args, **kwargs)

        return wrapped

    return patch
