# SPDX-License-Identifier: Apache-2.0
"""Configuration module for One Logger Training Telemetry."""
from typing import Callable, Optional, Union

from nv_one_logger.api.config import ApplicationType, OneLoggerConfig
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.utils import evaluate_value
from pydantic import model_validator

from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy


class TrainingTelemetryConfig(OneLoggerConfig):
    """Configuration for One Logger Training Telemetry.

    This class extends the base OneLoggerConfig with training-specific configuration options including
    world size, batch sizes, logging frequencies, and throughput-related settings.
    """

    # Note: Since this dataclass inherits from OneLoggerConfig, which has some fields with
    # defaults values, all the fields in this dataclass need to have default values.
    # If there is no reasonable default value for a field, we can use a dummy default and then
    # reject it in the __post_init__ method (see "world_size" and "global_batch_size" below).

    # This field is inhertied from the base class but has a default value in this subclass.
    app_type_or_fn: Union[ApplicationType, str, Callable[[], Union[str, ApplicationType]]] = ApplicationType.TRAINING

    # Number (or callable to get number) of processes participating in the training.
    world_size_or_fn: Union[int, Callable[[], int]] = 0

    @property
    def world_size(self) -> int:
        """Number of processes participating in the training."""
        return evaluate_value(self.world_size_or_fn)

    # Global batch size or function to compute it
    global_batch_size_or_fn: Union[int, Callable[[], int]] = 0

    @property
    def global_batch_size(self) -> int:
        """Global batch size."""
        return evaluate_value(self.global_batch_size_or_fn)

    # Whether to enable logging for the current rank in distributed training.
    enable_for_current_rank: bool = False

    # Frequency of logging, specified as the number of steps between logs. This knob
    # controls how frequently training progress is logged. The lower the value, the more frequently
    # training progress metrics are calculated and logged but the more data will be sent to the backends.
    log_every_n_train_iterations: int = 50

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

    # Flag (or callable to return flag) that whether to log throughput-related metrics
    is_log_throughput_enabled_or_fn: Union[bool, Callable[[], bool]] = False

    @property
    def is_log_throughput_enabled(self) -> bool:
        """Whether to log throughput-related metrics."""
        return evaluate_value(self.is_log_throughput_enabled_or_fn)

    # Size (or callable to generate the size) of each micro-batch in training (if applicable).
    micro_batch_size_or_fn: Optional[Union[int, Callable[[], int]]] = None

    @property
    def micro_batch_size(self) -> Optional[int]:
        """Size of each micro-batch in training."""
        return evaluate_value(self.micro_batch_size_or_fn)

    # Version (or callable to return version) of the data schema used for summarizing metrics.
    # If the schema of the data you collect changes over time, you can use this value to
    # keep track of which schema version is used for which run.
    summary_data_schema_version_or_fn: Union[str, Callable[[], str]] = "1.0.0"

    @property
    def summary_data_schema_version(self) -> str:
        """Version of the data schema used for summarizing metrics."""
        return evaluate_value(self.summary_data_schema_version_or_fn)

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

    # Strategy used for saving checkpoints
    save_checkpoint_strategy: CheckPointStrategy = CheckPointStrategy.SYNC

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

    @model_validator(mode="after")
    def validate_training_telemetry_config(self):
        """Validate the training telemetry configuration.

        This validator ensures that:
        - world_size is set to a non-zero value
        - global_batch_size is set to a non-zero value
        - flops_per_sample is set to a positive value when throughput logging is enabled

        Returns:
            TrainingTelemetryConfig: The validated configuration.

        Raises:
            OneLoggerError: If any required field is not set or if validation fails.
        """
        if self.world_size <= 0:
            raise OneLoggerError("world_size must be set to a non-zero value")
        if self.global_batch_size <= 0:
            raise OneLoggerError("global_batch_size must be set to a non-zero value")

        # Validate fields that are required only if throughput logging is enabled
        if self.is_log_throughput_enabled and (self.flops_per_sample is None or self.flops_per_sample <= 0):
            raise OneLoggerError("flops_per_sample must be set to a positive value when is_log_throughput_enabled is True")

        return self
