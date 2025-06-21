# SPDX-License-Identifier: Apache-2.0
"""Configuration for OneLogger."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

from strenum import StrEnum

from nv_one_logger.core.attributes import AttributeValue
from nv_one_logger.core.internal.utils import evaluate_value


class OneLoggerErrorHandlingStrategy(Enum):
    """Enum for the error handling strategy for OneLogger.

    This enum determines what happens when OneLogger encounters a fatal error (e.g., an exception in the instrumentation code or
    a problem with the OneLogger state). This does NOT affect handling of errors occuring when communicating with the telemetry backends
    (i.e., exporter failures, which are handled by the Recorder). Rather, this is about handling user errors when configuring OneLogger
    or bugs in telemetry code (e.g., assertion/invariant violations or hitting an inconsistent state).

    Our recommendation is to use PROPAGATE_EXCEPTIONS as it ensures that you get maximum visibility into the errors in the instrumentation code.
    But if you prefer to treat telemetry as a non-critical part of your application, you can use DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR.

    Read the docstrings for DefaultRecorder for more details on how errors from exporters (e.g., communication errors with telemetry
    backends) are handled when using that recorder.
    """

    """Propagate the exceptions to the caller.

    Use this strategy if you are OK with instrumentation exceptions to crash the application (or are willing to catch and handle such exceptions).
    Note that all of exceptions from One Logger will be instances of `OneLoggerError` or a subclass of it. This allows you to identify those exceptions
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


class ApplicationType(StrEnum):
    """Enum for common application types."""

    # Model Training (can include validation and testing of model)
    TRAINING = "training"

    # Model Validation (without training)
    VALIDATION = "validation"

    # Batch Inference (inference on a batch of data)
    BATCH_INFERENCE = "batch_inference"

    # Online Inference (inference on a single data point)
    ONLINE_INFERENCE = "online_inference"

    # Data Processing (e.g., ETL, ELT, data ingestion or data transformation pipelines)
    DATA_PROCESSING = "data_processing"


@dataclass
class LoggerConfig:
    """Configuration for how OneLogger logs its messages and errors."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    log_format: str = "%(name)s - %(levelname)s - %(message)s"

    """Path to the file where OneLogger INFO logs its messages."""
    log_file_path_for_info: Union[Path, str] = "onelogger.log"

    """Path to the file where OneLogger ERROR logs its messages."""
    log_file_path_for_err: Union[Path, str] = "onelogger.err"

    def __post_init__(self) -> None:
        """Validate that the log_level is one of the allowed values."""
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"log_level must be one of {'DEBUG', 'INFO', 'WARNING', 'ERROR'}, got {self.log_level}")


@dataclass
class OneLoggerConfig:
    """Configuration for OneLogger."""

    # The unique name for application. This name is used to identify the telemetry data related to various executions of
    # the same application in the OneLogger system (over time, across different machines/clusters, and across
    # different versions of the application).
    application_name: str

    # perf_tag or function to compute the perf tag. perf_tag is used to identify jobs whose performance is expected to be comparable.
    # Since this is a complex concept and is related to "session_tag", we strongly recommend that you read the "configuration"
    # section of README for more details.
    perf_tag_or_fn: Union[str, List[str], Callable[[], Union[str, List[str]]]]

    @property
    def perf_tag(self) -> Union[str, List[str]]:
        """Get the perf tag.

        Returns:
            Union[str, List[str]]: The evaluated perf tag value.
        """
        return evaluate_value(self.perf_tag_or_fn)  # type: ignore[return-value]

    # session_tag (or callable to generate the tag). session_tag is used to determine if two runs use the same code, config, and execution environment
    # (or the differences are not expected to impact the performance). Since this is a complex concept and is related to
    # "perf_tag", we strongly recommend that you read the "configuration" section of README for more details.
    session_tag_or_fn: Union[str, Callable[[], str]]

    @property
    def session_tag(self) -> str:
        """Get the session tag.

        Returns:
            str: The evaluated session tag value.
        """
        return evaluate_value(self.session_tag_or_fn)

    # This is used to classify the type of the application. We recommend using the ApplicationType enum rather than string
    # when possible to avoid accidental differences due to typos or casing differences.
    app_type_or_fn: Union[ApplicationType, str, Callable[[], Union[str, ApplicationType]]]

    @property
    def app_type(self) -> Union[ApplicationType, str]:
        """Get the application type.

        Returns:
            Union[ApplicationType, str]: The evaluated application type.
        """
        return evaluate_value(self.app_type_or_fn)

    # Flag (or callable to return flag) that indicates if this is a baseline run for comparison purposes.
    # A baseline run is a run that is used to set a performance baseline for future runs with the same
    # perf_tag.
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
    # By default, we will propagate the exceptions to the caller. This means that if there is an error in the instrumentation
    # code, it will crash the application.
    # You can choose a different strategy if you want to treat telemetry as a non-critical part of your application.
    # See the enum docstring for more details on each strategy and for our recommendations on how to set this value.
    error_handling_strategy: OneLoggerErrorHandlingStrategy = OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS

    # Flag to enable/disable OneLogger. If set to False, the library will not collect any telemetry data
    # and will act mostly as a no-op (except for a few lines of code that initialize the library and check for this flag).
    enable_one_logger: bool = True

    # Configuration for the logger used for logging messages and errors from the telemetry code.
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)

    def validate_config(self) -> None:
        """Validate the config.

        Note: we are not using __post_init__ because we cannot reliably honor the
            values set by user in error_handling_strategy while we are
            constructing the config object. So instead, we call the validation
            method explicitly in a context that can honor that setting.

        Raises:
        OneLoggerError: If any required field is not set or if validation fails.
        """
        pass
