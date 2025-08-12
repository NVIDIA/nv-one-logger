# SPDX-License-Identifier: Apache-2.0
"""Telemetry configuration protocol for OneLogger."""

from typing import Dict, Optional, Protocol, Union, runtime_checkable

from strenum import StrEnum

from nv_one_logger.core.attributes import AttributeValue


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


@runtime_checkable
class TelemetryConfig(Protocol):
    """Protocol for telemetry configuration.

    This protocol defines the interface that any telemetry configuration must implement.
    It allows the core module to reference telemetry configs without creating circular dependencies
    and provides type safety while allowing different implementations for different application types.

    The @runtime_checkable decorator enables runtime type checking with isinstance() and issubclass().
    Without this decorator, isinstance(obj, TelemetryConfig) would always return False, even for
    objects that structurally implement all the required methods and properties.

    Example usage:
        # This works because of @runtime_checkable
        if isinstance(some_config, TelemetryConfig):
            print("Valid telemetry config")

    Note: @runtime_checkable has some limitations - it only checks for the presence of methods
    and properties, not their signatures or return types. It's a structural check, not a
    behavioral one.
    """

    # Flag indicating whether the application has training iterations
    @property
    def is_train_iterations_enabled(self) -> bool:
        """Whether the application has training iterations that OneLogger should track metrics for."""
        ...

    # Flag indicating whether the application has validation/evaluation iterations
    @property
    def is_validation_iterations_enabled(self) -> bool:
        """Whether the application has validation/evaluation iterations that OneLogger should track metrics for."""
        ...

    # Flag indicating whether the application has test iterations
    @property
    def is_test_iterations_enabled(self) -> bool:
        """Whether the application has test iterations that OneLogger should track metrics for."""
        ...

    # Flag indicating whether the application saves checkpoints
    @property
    def is_save_checkpoint_enabled(self) -> bool:
        """Whether the application saves checkpoints that OneLogger should track metrics for."""
        ...

    # Application type for this telemetry configuration
    @property
    def app_type(self) -> Union[ApplicationType, str]:
        """Get the application type for this telemetry configuration."""
        ...

    # Custom metadata specific to telemetry. This metadata will be logged
    # as attributes of telemetry-related spans and events.
    @property
    def custom_metadata(self) -> Optional[Dict[str, AttributeValue]]:
        """Custom metadata specific to telemetry configuration."""
        ...
