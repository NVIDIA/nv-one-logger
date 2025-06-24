# SPDX-License-Identifier: Apache-2.0
"""Contains utilities for the core module."""

import os
from contextlib import contextmanager
from typing import Callable, Generator, Optional, TypeVar, Union


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
