# SPDX-License-Identifier: Apache-2.0
"""Contains utilities for safe execution of code.

This module provides utilities that help ensure that errors in onelogger library do not cause the instrumented application to crash.
"""

from functools import wraps
from typing import Any, Callable

from nv_one_logger.api.config import OneLoggerErrorHandlingStrategy
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.internal.logging import get_logger

_logger = get_logger(__name__)

# Depending on the user provided OneLoggerErrorHandlingStrategy, we may need to handle issues in collecting or exporting telemetry data
# in different ways. However, regardless of the strategy, we don't want to pollute the application logs with tens of errors related from telemetry code.
# This flag allows us to control how aggresively we log telemetry data issues to the application logs.
__telemetry_data_issue_already_logged = False


def exception_guard(func: Callable[..., Any]) -> Callable[..., Any]:  # noqa: C901
    """Handle exceptions from the wrapped function according to the configured OneLoggerErrorHandlingStrategy.

    This is a decorator function that wraps a given function to handle exceptions. All the telemetry code must be guarded with this decorator
    (directly or indirectly) so that the error handling strategy chosen by the user is respected.

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global __telemetry_data_issue_already_logged
        config = OneLoggerProvider.instance().config
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"Exception in telemetry code when calling {func.__name__}: {e}"
            try:
                if OneLoggerProvider.instance().one_logger_enabled:
                    # Inform all the exporters that the telemetry data has issues.
                    OneLoggerProvider.instance().recorder.telemetry_data_error(error_message)
            except Exception:
                pass  # not much we can do here.
            if config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS:
                _logger.error(error_message)
                raise e
            elif config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR:
                if OneLoggerProvider.instance().one_logger_enabled:
                    # Extra check to ensure that we only log the issue in the application logs only once.
                    # This is not normally needed as with the DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR strategy,
                    # we disable one logger after encourntering the first telemetry data error, which means there will
                    # be no further calls to one logger and no further telemetry data errors. But to be on the safe side
                    # we add this check to prevent filling the application logs with one logger errors even if
                    # disabling one logger doesn't take effect for some reason.
                    if not __telemetry_data_issue_already_logged:
                        __telemetry_data_issue_already_logged = True
                        _logger.error(f"OneLogger encountered a fatal error. Disabling OneLogger and telemetry data collection. Error: {error_message}")

                    OneLoggerProvider.instance().force_disable_logging()
            else:
                # This should never happen. If it does, it means you have added a new error handling strategy without updating this function.
                _logger.error(f"Unrecognized error handling strategy {config.error_handling_strategy}")
                raise e

    return wrapper


def safely_execute(func: Callable[..., Any]) -> Callable[..., Any]:
    """Execute the given callable safely.

    In particular, this decorator ensures that:
    - the wrapped function is executed only if one_logger is enabled.
    - applies exception_guard: any errors encountered during the execution of the wrapped function
      is handled according to the configured error handling strategy.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if OneLoggerProvider.instance().one_logger_enabled:
            return exception_guard(func)(*args, **kwargs)
        else:
            # Skipping execution because OneLogger is not enabled.
            _logger.warning(f"Skipping execution of {func.__name__} because OneLogger is not enabled.")
            return None

    return wrapper
