# SPDX-License-Identifier: Apache-2.0
from typing import Any


class OneLoggerError(Exception):
    """
    Represents an error occurring in the one logger library.

    This exception is raised when there is an error in the internal state (e.g., due to incorrect usage of the one logger library).
    Note that the config of the library has knobs to control the behavior of the library when an error occurs (propagate this error
    to the application or log and silently suppress it). See OneLoggerErrorHandlingStrategy.

    ************** NOTE **************
    If you are contributing to the One Logger source code, make sure all the exceptions you raise are instances of this class or its
    source code, make sure all the exceptions you raise are instances of this class or its
    subclasses. See the comments on OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS to understand why.
    """

    def __init__(self, message: str):
        super().__init__(message)


def assert_that(condition: Any, message: str) -> None:
    """Safely asserts that a condition is true.

    In general, observability code must not cause the application to crash. Because of this, instead of using an assert,
    we use this function to check invariants. If an invariant is violated, we log an error and throw a OneLoggerError.
    Since we have code to catch OneLoggerErrors and suppress them, this will not cause the application to crash.
    """
    if not condition:
        raise OneLoggerError(message)
