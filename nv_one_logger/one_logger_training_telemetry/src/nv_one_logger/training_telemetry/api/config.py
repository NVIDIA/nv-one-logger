# SPDX-License-Identifier: Apache-2.0
"""Configuration module for One Logger Training Telemetry."""
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from nv_one_logger.api.telemetry_config import ApplicationType
from nv_one_logger.core.attributes import AttributeValue
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.utils import evaluate_value
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy


class TrainingTelemetryConfig(BaseModel):
    """Configuration for training telemetry specific settings.

    This class implements the TelemetryConfig protocol and contains all training-specific
    configuration options. It includes all the base telemetry fields plus training-specific ones.
    """

    # This class implements the TelemetryConfig protocol
    # The @runtime_checkable decorator on TelemetryConfig allows isinstance() checks

    # Base telemetry configuration fields (from TelemetryConfig protocol)

    # Flag (or callable to return flag) that whether to log training iterations
    is_train_iterations_enabled_or_fn: Union[bool, Callable[[], bool]] = True

    @property
    def is_train_iterations_enabled(self) -> bool:
        """Whether to log training iterations."""
        return evaluate_value(self.is_train_iterations_enabled_or_fn)

    # Flag (or callable to return flag) that whether to log eval/validation iterations
    is_validation_iterations_enabled_or_fn: Union[bool, Callable[[], bool]] = True

    @property
    def is_validation_iterations_enabled(self) -> bool:
        """Whether to log eval/validation iterations."""
        return evaluate_value(self.is_validation_iterations_enabled_or_fn)

    # Flag (or callable to return flag) that whether to log test iterations
    is_test_iterations_enabled_or_fn: Union[bool, Callable[[], bool]] = True

    @property
    def is_test_iterations_enabled(self) -> bool:
        """Whether to log test iterations."""
        return evaluate_value(self.is_test_iterations_enabled_or_fn)

    # Flag (or callable to return flag) that whether to log metrics related to saving checkpoints
    is_save_checkpoint_enabled_or_fn: Union[bool, Callable[[], bool]] = True

    @property
    def is_save_checkpoint_enabled(self) -> bool:
        """Whether to log metrics related to saving checkpoints."""
        return evaluate_value(self.is_save_checkpoint_enabled_or_fn)

    @property
    def app_type(self) -> Union[ApplicationType, str]:
        """Application type for this telemetry configuration."""
        return ApplicationType.TRAINING

    # Custom metadata specific to telemetry. This metadata will be logged
    # as attributes of telemetry-related spans and events.
    custom_metadata: Optional[Dict[str, AttributeValue]] = None

    # Training-specific configuration fields

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

    # Global batch size or function to compute it
    global_batch_size_or_fn: Union[int, Callable[[], int]]

    @property
    def global_batch_size(self) -> int:
        """Global batch size."""
        return evaluate_value(self.global_batch_size_or_fn)

    # Size (or callable to generate the size) of each micro-batch in training (if applicable).
    micro_batch_size_or_fn: Optional[Union[int, Callable[[], int]]] = None

    @property
    def micro_batch_size(self) -> Optional[int]:
        """Size of each micro-batch in training."""
        return evaluate_value(self.micro_batch_size_or_fn)

    # Sequence length of a training sample or function to calculate the length (if applicable).
    seq_length_or_fn: Optional[Union[int, Callable[[], int]]] = None

    @property
    def seq_length(self) -> Optional[int]:
        """Sequence length of a training sample."""
        return evaluate_value(self.seq_length_or_fn)

    # FLOPs per sample or function to compute FLOPs per sample.
    # NOTE: this must be set if `is_log_throughput_enabled` is set to `True`.
    flops_per_sample_or_fn: Optional[Union[int, Callable[[], int]]] = None

    @property
    def flops_per_sample(self) -> Optional[int]:
        """FLOPS per sample."""
        return evaluate_value(self.flops_per_sample_or_fn)

    # Target number of training iterations or callable to generate it.
    # This is used to calculate the training throughput.
    train_iterations_target_or_fn: Optional[Union[int, Callable[[], int]]] = None

    @property
    def train_iterations_target(self) -> Optional[int]:
        """Target number of training iterations."""
        return evaluate_value(self.train_iterations_target_or_fn)

    # Target number of training samples or function to generate the number
    # This is used to calculate the training throughput.
    train_samples_target_or_fn: Optional[Union[int, Callable[[], int]]] = None

    @property
    def train_samples_target(self) -> Optional[int]:
        """Target number of training samples."""
        return evaluate_value(self.train_samples_target_or_fn)

    # Frequency of logging, specified as the number of steps between logs. This knob
    # controls how frequently training progress is logged. The lower the value, the more frequently
    # training progress metrics are calculated and logged but the more data will be sent to the backends.
    log_every_n_train_iterations: int = 50

    # Flag (or callable to return flag) that whether to log throughput-related metrics
    is_log_throughput_enabled_or_fn: Union[bool, Callable[[], bool]] = False

    @property
    def is_log_throughput_enabled(self) -> bool:
        """Whether to log throughput-related metrics."""
        return evaluate_value(self.is_log_throughput_enabled_or_fn)

    # Strategy used for saving checkpoints
    save_checkpoint_strategy: CheckPointStrategy = CheckPointStrategy.SYNC

    @model_validator(mode="after")
    def validate_training_telemetry_config(self) -> Self:
        """Validate the training telemetry configuration.

        This validator ensures that:
        - global_batch_size is set to a positive value
        - log_every_n_train_iterations is set to a positive value
        - flops_per_sample is set to a positive value when throughput logging is enabled

        Returns:
            TrainingTelemetryConfig: The validated configuration.

        Raises:
            OneLoggerError: If any required field is not set or if validation fails.
        """
        if self.global_batch_size <= 0:
            raise OneLoggerError("global_batch_size must be set to a positive value")

        # Validate log_every_n_train_iterations is positive
        if self.log_every_n_train_iterations <= 0:
            raise OneLoggerError("log_every_n_train_iterations must be set to a positive value ")

        # Validate fields that are required only if throughput logging is enabled
        if self.is_log_throughput_enabled and (self.flops_per_sample is None or self.flops_per_sample <= 0):
            raise OneLoggerError("flops_per_sample must be set to a positive value when is_log_throughput_enabled is True")

        return self
