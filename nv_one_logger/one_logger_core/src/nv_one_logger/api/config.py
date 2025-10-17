# SPDX-License-Identifier: Apache-2.0
"""Configuration for OneLogger."""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from nv_one_logger.api.telemetry_config import TelemetryConfig
from nv_one_logger.core.attributes import AttributeValue
from nv_one_logger.core.internal.utils import evaluate_value

# Note: We use plain Python logging here instead of OneLogger's get_logger() to avoid circular dependencies.
# The logging module (nv_one_logger.core.internal.logging) imports from this config module,
# so we cannot import get_logger here. This logger is used for configuration validation messages
# and will always log regardless of OneLogger's log_level settings, which is appropriate for
# important configuration warnings.
logger = logging.getLogger(__name__)

# Ensure the logger has a handler to display messages even if the application hasn't configured logging yet
# This is important because config validation happens early in the initialization process
# if not logger.handlers:
#    _handler = logging.StreamHandler()
#    _handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
#    logger.addHandler(_handler)
#    logger.setLevel(logging.INFO)


class OneLoggerErrorHandlingStrategy(Enum):
    """Enum for the error handling strategy for OneLogger.

    This enum determines what happens when OneLogger encounters a fatal error (e.g., an exception in the instrumentation code or
    a problem with the OneLogger state). This does NOT affect handling of errors occuring when communicating with the telemetry backends
    (i.e., exporter failures, which are handled by the Recorder). Rather, this is about handling user errors when configuring OneLogger
    or bugs in telemetry code (e.g., assertion/invariant violations or hitting an inconsistent state).

    Our recommendation is to use PROPAGATE_EXCEPTIONS for development scenarios where you want maximum visibility into errors,
    or DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for production scenarios where training should continue even if logging fails.

    Note: If you don't explicitly specify a strategy, OneLogger will automatically apply smart defaults:
    - Enabled ranks (enable_for_current_rank=True): PROPAGATE_EXCEPTIONS (for error visibility)
    - Disabled ranks (enable_for_current_rank=False): DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR (for safety)

    Read the docstrings for DefaultRecorder for more details on how errors from exporters (e.g., communication errors with telemetry
    backends) are handled when using that recorder.
    """

    """Propagate the exceptions to the caller.

    Use this strategy if you are OK with instrumentation exceptions to crash the application (or are willing to catch and handle such exceptions).
    Note that all of exceptions from One Logger will be instances of `OneLoggerError` or a subclass of it. This allows you to identify those exceptions  # noqa: E501
    and react to them accordingly.
    The advantage of this strategy is that you get maximum visibility into the errors in the instrumentation code.
    """
    PROPAGATE_EXCEPTIONS = "propagate_exceptions"

    """Disable OneLogger silently and report metric errors to the telemetry backends.

    With this option, if instrumentation code encounters any errors, the library catches/suppresses the exception letting the
    application continue running, logs a single error to the application logs, and informs the telemtry backends that
    the telemetry data has errors.
    """
    DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR = "disable_quietly_and_report_metric_error"


class LoggerConfig(BaseModel):
    """Configuration for how OneLogger logs its messages and errors."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    log_format: str = "%(name)s - %(levelname)s - %(message)s"

    # Path to the file where OneLogger INFO logs its messages.
    log_file_path_for_info: Union[Path, str] = "onelogger.log"

    # Path to the file where OneLogger ERROR logs its messages.
    log_file_path_for_err: Union[Path, str] = "onelogger.err"

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate the logger config."""
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level must be one of {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}, got {self.log_level}")
        return self


class OneLoggerConfig(BaseModel):
    """Configuration for OneLogger."""

    # Pydantic model configuration to allow arbitrary types in field definitions.
    # This is necessary because the telemetry_config field uses a Protocol type (TelemetryConfig),
    # which is a custom type that Pydantic cannot automatically generate a schema for.
    # By setting arbitrary_types_allowed=True,
    # we tell Pydantic to accept any type for this field and skip schema validation for it.
    #
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The unique name for application. This name is used to identify the telemetry data related to various executions of
    # the same application in the OneLogger system (over time, across different machines/clusters, and across
    # different versions of the application).
    application_name: str

    # Number of processes participating in the telemetry collection.
    # This is a fundamental configuration that affects the entire application, not just training telemetry.
    world_size_or_fn: Union[int, Callable[[], int]]

    @property
    def world_size(self) -> int:
        """Number of processes participating in the telemetry collection."""
        return evaluate_value(self.world_size_or_fn)

    # session_tag (or callable to generate the tag). session_tag is used to used to identify jobs that together contribute
    # to the same task. This means the jobs are "logically" part of a single larger job (e.g.,
    # a long running job that is split into multiple jobs due to resource constraints or resuming after a failure).
    session_tag_or_fn: Union[str, Callable[[], str]]

    @property
    def session_tag(self) -> str:
        """Get the session tag.

        Returns:
            str: The evaluated session tag value.
        """
        return evaluate_value(self.session_tag_or_fn)

    # Flag (or callable to return flag) that indicates if this is a baseline run for comparison purposes.
    # A baseline run is a run that is used to set a performance baseline for future runs.
    is_baseline_run_or_fn: Union[bool, Callable[[], bool]] = False

    @property
    def is_baseline_run(self) -> bool:
        """Get the baseline run flag.

        Returns:
            bool: The evaluated baseline run flag value.
        """
        return evaluate_value(self.is_baseline_run_or_fn)

    # Custom metadata to be logged with the training telemetry data.
    # This metadata will be logged as-is, without any modification as an
    # attribute of the APPLICATION span.
    custom_metadata: Optional[Dict[str, AttributeValue]] = None

    # The strategy to use for handling errors in the instrumentation code.
    # If not explicitly provided, OneLogger applies defaults based on rank enablement:
    # - Enabled ranks (where OneLogger is active): Will propagate errors
    # - Disabled ranks (where OneLogger is inactive): Will suppress errors quietly
    # You can explicitly set a strategy to override this automatic behavior.
    # See the enum docstring for more details on each strategy and for our recommendations on how to set this value.
    error_handling_strategy: OneLoggerErrorHandlingStrategy = OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS

    # Whether to enable logging for the current rank in distributed training.
    # This controls whether OneLogger is enabled for the current process/rank.
    # In distributed training scenarios, you can set this to False for ranks where you don't want logging.
    enable_for_current_rank: bool = True

    # Configuration for the logger used for logging messages and errors from the telemetry code.
    logger_config: LoggerConfig = LoggerConfig()

    # Version (or callable to return version) of the data schema used for summarizing metrics.
    # If the schema of the data you collect changes over time, you can use this value to
    # keep track of which schema version is used for which run.
    summary_data_schema_version: Union[str, str] = "1.0.0"

    # Telemetry-specific configuration.
    # This field contains telemetry-specific settings and parameters.
    # It is optional and can be None if no telemetry is needed.
    #
    # Note: This field uses the TelemetryConfig Protocol type, which is why we need
    # arbitrary_types_allowed=True in the model_config above. The actual value must be
    # a concrete class that implements the TelemetryConfig protocol interface.
    telemetry_config: Optional[TelemetryConfig] = None

    @model_validator(mode="before")
    @classmethod
    def apply_error_handling_defaults(cls, data):
        """Apply defaults for error handling strategy based on rank enablement.

        This applies production-safe defaults by automatically using quiet error handling
        for disabled ranks when the user hasn't explicitly specified an error handling strategy.
        """
        if isinstance(data, dict):
            # Check if user explicitly provided error_handling_strategy
            user_set_error_handling = "error_handling_strategy" in data

            # Apply automatic quiet error handling for disabled ranks when not explicitly set by user
            if not user_set_error_handling and not data.get("enable_for_current_rank", True):  # Default to True if not specified
                rank = int(os.environ.get("RANK", 0))
                logger.warning(
                    f"OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR "
                    f"for rank (rank={rank}) with OneLogger disabled. "
                    "To override: explicitly set error_handling_strategy parameter."
                )

                # Set quiet error handling for production safety
                data["error_handling_strategy"] = OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR

                # Set log level to CRITICAL to suppress further OneLogger logs for disabled ranks
                if "logger_config" not in data:
                    data["logger_config"] = {}
                if isinstance(data["logger_config"], dict) and "log_level" not in data["logger_config"]:
                    data["logger_config"]["log_level"] = "CRITICAL"

        return data

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate the OneLogger configuration.

        This validator ensures that:
        - application_name is not empty
        - world_size is set to a positive value
        - custom_metadata keys are valid strings (if provided)
        - telemetry_config implements the TelemetryConfig protocol (if provided)

        Returns:
            OneLoggerConfig: The validated configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.application_name or not self.application_name.strip():
            raise ValueError("application_name cannot be empty or whitespace-only")

        if self.world_size <= 0:
            raise ValueError("world_size must be set to a positive value")

        if self.custom_metadata is not None:
            for key in self.custom_metadata.keys():
                if not key.strip():
                    raise ValueError("custom_metadata keys must be non-empty strings")

        return self
