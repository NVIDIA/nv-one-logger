# SPDX-License-Identifier: Apache-2.0
"""Module for training telemetry attributes.

This module contains classes for various attributes used in training telemetry.
"""
# pyright: reportUnnecessaryComparison=false

from typing import Dict, List, Optional, Union

from nv_one_logger.core.attributes import Attributes, AttributeValue
from nv_one_logger.core.exceptions import assert_that

from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy

# **********************************************************************************************************************
# * Span Attributes
# **********************************************************************************************************************


class TrainingLoopAttributes(Attributes):
    """Span attributes for the spans of type TRAINING_LOOP span for training jobs."""

    @classmethod
    def create(
        cls,
        train_iterations_start: int,
        train_samples_start: int,
        train_iterations_target: Optional[int] = None,
        train_samples_target: Optional[int] = None,
        train_tokens_target: Optional[int] = None,
    ) -> "TrainingLoopAttributes":
        """Create a TrainingLoopAttributes object.

        Args:
            train_iterations_start: The starting iteration number / global step(could be non-zero if the job loads a checkpoint and starts from there).
            train_samples_start: The starting sample number (could be non-zero if the job loads a checkpoint and starts from there).
            train_iterations_target: Target number of training iterations.
            train_samples_target: Target number of training samples.
            train_tokens_target: Target numbrer of training tokens.
        """
        attributes = cls()
        assert_that(train_iterations_start is not None, "train_iterations_start is required.")
        assert_that(train_samples_start is not None, "train_samples_start is required.")
        attributes.add("train_iterations_start", train_iterations_start)
        attributes.add("train_samples_start", train_samples_start)
        if train_iterations_target is not None:
            attributes.add("train_iterations_target", train_iterations_target)
        if train_samples_target is not None:
            attributes.add("train_samples_target", train_samples_target)
        if train_tokens_target is not None:
            attributes.add("train_tokens_target", train_tokens_target)
        return attributes

    @property
    def train_iterations_start(self) -> int:
        """The starting iteration number/global step (could be non-zero if the job loads a checkpoint and starts from there)."""
        val = self.get_int_value("train_iterations_start")
        assert_that(val is not None, "train_iterations_start is required.")
        return val  # type: ignore

    @property
    def train_samples_start(self) -> int:
        """The starting sample number (could be non-zero if the job loads a checkpoint and starts from there)."""
        val = self.get_int_value("train_samples_start")
        assert_that(val is not None, "train_samples_start is required.")
        return val  # type: ignore

    @property
    def train_iterations_target(self) -> Optional[int]:
        """Target number of training iterations."""
        return self.get_int_value("train_iterations_target")

    @property
    def train_samples_target(self) -> Optional[int]:
        """Target number of training samples."""
        return self.get_int_value("train_samples_target")

    @property
    def train_tokens_target(self) -> Optional[int]:
        """Target number of training tokens."""
        return self.get_int_value("train_tokens_target")


class CheckpointSaveSpanAttributes(Attributes):
    """Span attributes for the spans of type CHECKPOINT_SAVE_ASYNC or CHECKPOINT_SAVE_SYNC for training jobs."""

    @classmethod
    def create(cls, checkpoint_strategy: CheckPointStrategy, current_iteration: int, save_checkpoint_attempt_count: int) -> "CheckpointSaveSpanAttributes":
        """Create a CheckpointSaveSpanAttributes object.

        Args:
            checkpoint_strategy: The strategy used for saving checkpoints (SYNC or ASYNC).
            current_iteration: The current iteration number as of the time the checkpont save started.
            save_checkpoint_attempt_count: Return the number of times checkpoint save operation was attempted so far
            (includes failed attempts as well as the current save attempt).
        """
        attributes = cls()
        assert_that(checkpoint_strategy is not None, "checkpoint_strategy is required.")
        assert_that(current_iteration is not None, "current_iteration is required.")
        assert_that(save_checkpoint_attempt_count is not None, "save_checkpoint_attempt_count is required.")
        attributes.add("checkpoint_strategy", checkpoint_strategy)
        attributes.add("current_iteration", current_iteration)
        attributes.add("save_checkpoint_attempt_count", save_checkpoint_attempt_count)
        return attributes

    @property
    def checkpoint_strategy(self) -> CheckPointStrategy:
        """The strategy used for saving checkpoints (SYNC or ASYNC)."""
        val = self.get_str_value("checkpoint_strategy")
        assert_that(val is not None, "checkpoint_strategy is required.")
        return val  # type: ignore

    @property
    def current_iteration(self) -> int:
        """The current iteration number as of the time the checkpont save started."""
        val = self.get_int_value("current_iteration")
        assert_that(val is not None, "current_iteration is required.")
        return val  # type: ignore

    @property
    def save_checkpoint_attempt_count(self) -> int:
        """The number of times checkpoint save operation was attempted so far.

        This includes failed attempts as well as the current save attempt.
        """
        val = self.get_int_value("save_checkpoint_attempt_count")
        assert_that(val is not None, "save_checkpoint_attempt_count is required.")
        return val  # type: ignore


# **********************************************************************************************************************
# * Event Attributes
# **********************************************************************************************************************


# We keep track of start and end time for all spans as well as the timestamp for all events. But in training jobs, for certain
# spans and events, we collect more attributes beyond the timing data. Below, we have a list of attributes for such spans and events.
# Note that the following attributes are standard attributes that the library collects automatically but the users of the library are
# free to collect and report additional attributes and associate them with the spans and events.
class OneLoggerInitializationAttributes(Attributes):
    """Common attributes for the event of ONE_LOGGER_INITIALIZATION of the APPLICATION span for training jobs."""

    @classmethod
    def create(
        cls,
        one_logger_training_telemetry_version: str,
        enable_for_current_rank: bool,
        perf_tag: Union[str, List[str]],
        session_tag: str,
        app_type: str,
        log_every_n_train_iterations: int,
        world_size: int,
        global_batch_size: int,
        is_baseline_run: bool,
        is_train_iterations_enabled: bool,
        is_validation_iterations_enabled: bool,
        is_test_iterations_enabled: bool,
        is_save_checkpoint_enabled: bool,
        is_log_throughput_enabled: bool,
        summary_data_schema_version: str,
        node_name: str,
        rank: int,
        checkpoint_strategy: CheckPointStrategy,
        micro_batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        custom_metadata: Optional[Dict[str, AttributeValue]] = None,
    ) -> "OneLoggerInitializationAttributes":
        """Create a OneLoggerInitializationAttributes object.

        Args:
            one_logger_training_telemetry_version: Version of the one-logger-training-telemetry package.
            enable_for_current_rank: Whether to enable logging for the current rank in distributed training.
            perf_tag: Used to identify jobs whose performance is expected to be comparable.
            session_tag: Used to determine if two runs use the same code, config, and execution environment.
            app_type: Type of the application run (e.g., training, validation).
            log_every_n_train_iterations: Frequency of logging, specified as the number of steps between logs.
            world_size: Number of processes participating in the training.
            global_batch_size: Global batch size for training.
            is_baseline_run: Flag that indicates if this is a baseline run for comparison purposes.
            is_train_iterations_enabled: Whether to log training iterations.
            is_validation_iterations_enabled: Whether to log eval/validation iterations.
            is_test_iterations_enabled: Whether to log test iterations.
            is_save_checkpoint_enabled: Whether to log metrics related to saving checkpoints.
            is_log_throughput_enabled: Whether to log throughput-related metrics.
            summary_data_schema_version: Version of the data schema used for summarizing metrics.
            node_name: Name of the node (hostname).
            rank: The rank of the current process in distributed training.
            checkpoint_strategy: Strategy used for saving checkpoints.
            micro_batch_size: Size of each micro-batch in training (if applicable).
            seq_length: Sequence length of a training sample (if applicable).
            custom_metadata: Custom metadata to be logged with the training telemetry data. The metadata dictionary
            will be flattened to a string list of the form ["key1:value1", "key2:value2",..."].
        """
        attributes = cls()
        assert_that(one_logger_training_telemetry_version is not None, "one_logger_training_telemetry_version is required.")
        assert_that(enable_for_current_rank is not None, "enable_for_current_rank is required.")
        assert_that(perf_tag is not None, "perf_tag is required.")
        assert_that(session_tag is not None, "session_tag is required.")
        assert_that(app_type is not None, "app_type is required.")
        assert_that(log_every_n_train_iterations is not None, "log_every_n_train_iterations is required.")
        assert_that(world_size is not None, "world_size is required.")
        assert_that(global_batch_size is not None, "global_batch_size is required.")
        assert_that(is_baseline_run is not None, "is_baseline_run is required.")
        assert_that(is_train_iterations_enabled is not None, "is_train_iterations_enabled is required.")
        assert_that(is_validation_iterations_enabled is not None, "is_validation_iterations_enabled is required.")
        assert_that(is_test_iterations_enabled is not None, "is_test_iterations_enabled is required.")
        assert_that(is_save_checkpoint_enabled is not None, "is_save_checkpoint_enabled is required.")
        assert_that(is_log_throughput_enabled is not None, "is_log_throughput_enabled is required.")
        assert_that(summary_data_schema_version is not None, "summary_data_schema_version is required.")
        assert_that(node_name is not None, "node_name is required.")
        assert_that(rank is not None, "rank is required.")
        assert_that(checkpoint_strategy is not None, "checkpoint_strategy is required.")

        attributes.add("one_logger_training_telemetry_version", one_logger_training_telemetry_version)
        attributes.add("enable_for_current_rank", enable_for_current_rank)
        attributes.add("perf_tag", perf_tag)  # type: ignore[reportArgumentType]
        attributes.add("session_tag", session_tag)
        attributes.add("app_type", app_type)
        attributes.add("log_every_n_train_iterations", log_every_n_train_iterations)
        attributes.add("world_size", world_size)
        attributes.add("global_batch_size", global_batch_size)
        attributes.add("is_baseline_run", is_baseline_run)
        attributes.add("is_train_iterations_enabled", is_train_iterations_enabled)
        attributes.add("is_validation_iterations_enabled", is_validation_iterations_enabled)
        attributes.add("is_test_iterations_enabled", is_test_iterations_enabled)
        attributes.add("is_save_checkpoint_enabled", is_save_checkpoint_enabled)
        attributes.add("is_log_throughput_enabled", is_log_throughput_enabled)
        attributes.add("summary_data_schema_version", summary_data_schema_version)
        attributes.add("node_name", node_name)
        attributes.add("rank", rank)
        attributes.add("checkpoint_strategy", checkpoint_strategy)
        if micro_batch_size is not None:
            attributes.add("micro_batch_size", micro_batch_size)
        if seq_length is not None:
            attributes.add("seq_length", seq_length)
        if custom_metadata is not None:
            # Flatten custom metadata to comply with the expected type of the `add` method.
            attributes.add("custom_metadata", [f"{k}:{v}" for k, v in custom_metadata.items()])

        return attributes

    @property
    def one_logger_training_telemetry_version(self) -> str:
        """Version of the one-logger-training-telemetry package."""
        val = self.get_str_value("one_logger_training_telemetry_version")
        assert_that(val is not None, "one_logger_training_telemetry_version is required.")
        return val  # type: ignore

    @property
    def enable_for_current_rank(self) -> bool:
        """Whether to enable logging for the current rank in distributed training."""
        val = self.get_bool_value("enable_for_current_rank")
        assert_that(val is not None, "enable_for_current_rank is required.")
        return val  # type: ignore

    @property
    def perf_tag(self) -> Union[str, List[str]]:
        """Used to identify jobs whose performance is expected to be comparable."""
        val = self.get_str_value("perf_tag")
        assert_that(val is not None, "perf_tag is required.")
        return val  # type: ignore

    @property
    def session_tag(self) -> str:
        """Used to determine if two runs use the same code, config, and execution environment."""
        val = self.get_str_value("session_tag")
        assert_that(val is not None, "session_tag is required.")
        return val  # type: ignore

    @property
    def app_type(self) -> str:
        """Type of the application run (e.g., training, validation)."""
        val = self.get_str_value("app_type")
        assert_that(val is not None, "app_type is required.")
        return val  # type: ignore

    @property
    def log_every_n_train_iterations(self) -> int:
        """Frequency of logging, specified as the number of steps between logs."""
        val = self.get_int_value("log_every_n_train_iterations")
        assert_that(val is not None, "log_every_n_train_iterations is required.")
        return val  # type: ignore

    @property
    def world_size(self) -> int:
        """Number of processes participating in the training."""
        val = self.get_int_value("world_size")
        assert_that(val is not None, "world_size is required.")
        return val  # type: ignore

    @property
    def global_batch_size(self) -> int:
        """Global batch size for training."""
        val = self.get_int_value("global_batch_size")
        assert_that(val is not None, "global_batch_size is required.")
        return val  # type: ignore

    @property
    def is_baseline_run(self) -> bool:
        """Flag that indicates if this is a baseline run for comparison purposes."""
        val = self.get_bool_value("is_baseline_run")
        assert_that(val is not None, "is_baseline_run is required.")
        return val  # type: ignore

    @property
    def is_train_iterations_enabled(self) -> bool:
        """Whether to log training iterations."""
        val = self.get_bool_value("is_train_iterations_enabled")
        assert_that(val is not None, "is_train_iterations_enabled is required.")
        return val  # type: ignore

    @property
    def is_validation_iterations_enabled(self) -> bool:
        """Whether to log eval/validation iterations."""
        val = self.get_bool_value("is_validation_iterations_enabled")
        assert_that(val is not None, "is_validation_iterations_enabled is required.")
        return val  # type: ignore

    @property
    def is_test_iterations_enabled(self) -> bool:
        """Whether to log test iterations."""
        val = self.get_bool_value("is_test_iterations_enabled")
        assert_that(val is not None, "is_test_iterations_enabled is required.")
        return val  # type: ignore

    @property
    def is_save_checkpoint_enabled(self) -> bool:
        """Whether to log metrics related to saving checkpoints."""
        val = self.get_bool_value("is_save_checkpoint_enabled")
        assert_that(val is not None, "is_save_checkpoint_enabled is required.")
        return val  # type: ignore

    @property
    def is_log_throughput_enabled(self) -> bool:
        """Whether to log throughput-related metrics."""
        val = self.get_bool_value("is_log_throughput_enabled")
        assert_that(val is not None, "is_log_throughput_enabled is required.")
        return val  # type: ignore

    @property
    def summary_data_schema_version(self) -> str:
        """Version of the data schema used for summarizing metrics."""
        val = self.get_str_value("summary_data_schema_version")
        assert_that(val is not None, "summary_data_schema_version is required.")
        return val  # type: ignore

    @property
    def node_name(self) -> str:
        """Name of the node (hostname)."""
        val = self.get_str_value("node_name")
        assert_that(val is not None, "node_name is required.")
        return val  # type: ignore

    @property
    def rank(self) -> int:
        """The rank of the current process in distributed training."""
        val = self.get_int_value("rank")
        assert_that(val is not None, "rank is required.")
        return val  # type: ignore

    @property
    def checkpoint_strategy(self) -> CheckPointStrategy:
        """Strategy used for saving checkpoints."""
        val = self.get_str_value("checkpoint_strategy")
        assert_that(val is not None, "checkpoint_strategy is required.")
        return val  # type: ignore

    @property
    def micro_batch_size(self) -> Optional[int]:
        """Size of each micro-batch in training (if applicable)."""
        return self.get_int_value("micro_batch_size")

    @property
    def seq_length(self) -> Optional[int]:
        """Sequence length of a training sample (if applicable)."""
        return self.get_int_value("seq_length")

    @property
    def custom_metadata(self) -> Optional[List[str]]:
        """Custom metadata to be logged with the training telemetry data."""
        if "custom_metadata" not in self.keys():
            return None
        return self["custom_metadata"].value  # type: ignore[reportReturnType]


class TrainingMetricsUpdateAttributes(Attributes):
    """Event attributes for a TRAINING_MULTI_ITERATION_METRICS_UPDATE event of the TRAINING_LOOP span for training jobs.

    These attributes contain the metrics aggregated over a window of N iterations (N == TrainingMetricsUpdateAttributes.num_iterations).
    """

    @classmethod
    def create(  # noqa: C901
        cls,
        train_iterations_start: int,
        current_iteration: int,
        num_iterations: int,
        train_samples_start: int,
        num_train_samples: int,
        interval: int,
        avg_iteration_time_sec: float,
        min_iteration_time_sec: float,
        max_iteration_time_sec: float,
        total_iteration_time_sec: float,
        avg_forward_time_sec: Optional[float] = None,
        avg_backward_time_sec: Optional[float] = None,
        avg_dataloader_time_sec: Optional[float] = None,
        avg_tflops: Optional[float] = None,
        train_tokens: Optional[int] = None,
        avg_tokens_per_second: Optional[float] = None,
        latest_loss: Optional[float] = None,
        avg_batch_size: Optional[int] = None,
        completed_floating_point_operations_overall: Optional[int] = None,
        total_flops: Optional[int] = None,
        train_throughput_per_gpu: Optional[float] = None,
        train_throughput_per_gpu_max: Optional[float] = None,
        train_throughput_per_gpu_min: Optional[float] = None,
    ) -> "TrainingMetricsUpdateAttributes":
        """Create a TrainingMetricsUpdateAttributes object.

        Args:
            train_iterations_start: The starting iteration number / global step(could be non-zero
                if the job loads a checkpoint and starts from there).
            current_iteration: The current iteration number at the time the event is generated
                (includes the iterations from the loaded checkpoint, if any).
            num_iterations: The number of iterations whose metrics are aggregated in this event, which
                is the same as the number of iterations performed in the current job. All the aggregated metrics
                below (avg_xxx, max_xxx, min_xxx, total_xxx) are computed over these iterations.
            train_samples_start: The starting sample number (could be non-zero if the job loads a checkpoint and starts from there).
                This corresponds to the "train_iterations_start" attribute.
            num_train_samples: The number of samples processed during the "num_iterations", which
                is the same as the number of samples used for training in the current job.
            interval: The interval between the current and previous iteration where a similar
                event was reported. That is, the number of training iterations between the last
                TRAINING_MULTI_ITERATION_METRICS_UPDATE event and this one.
            avg_iteration_time_sec: The average iteration time for the current job in seconds. None if unknown or unmeasured.
            min_iteration_time_sec: The minimum iteration time for the current job in seconds. None if unknown or unmeasured.
            max_iteration_time_sec: The maximum iteration time for the current job in seconds. None if unknown or unmeasured.
            total_iteration_time_sec: The total iteration time for the current job in seconds. None if unknown or unmeasured.
            avg_forward_time_sec: The average forward time for the current job in seconds. None if unknown or unmeasured.
            avg_backward_time_sec: The average backward time for the current job in seconds. None if unknown or unmeasured.
            avg_dataloader_time_sec: The average dataloader time for the current job in seconds. None if unknown or unmeasured.
            avg_tflops: The avg_tflops (tera flops) for the current job. None if unknown or unmeasured.
            train_tokens: Number of training tokens processed so far in the current job. None if unknown or unmeasured.
            avg_tokens_per_second: The tokens per second for the current job. None if unknown or unmeasured.
            latest_loss: The latest_loss. None if unknown or unmeasured.
            avg_batch_size: The average batch size for the current job. None if unknown or unmeasured.
            completed_floating_point_operations_overall: Number of floating point operations completed
                so far (including the ones from the loaded checkpoint and the ones from the current job).
                None if unknown or unmeasured.
            train_throughput_per_gpu: The train throughput per GPU during the current job in tflops
                (one trillion floating point operations per second). This is the average over the job so far.
                None if unknown or unmeasured.
            train_throughput_per_gpu_max: The max train throughput per GPU during the current job in tflops
                (one trillion floating point operations per second).  This value is computed as the max of
                the per-iteration train throughput values (this is the throughput of the iteration with the
                highest throughput).  None if unknown or unmeasured.
            train_throughput_per_gpu_min: The min train throughput per GPU during the current job in tflops
                (one trillion floating point operations per second). This value is computed as the min of the
                per-iteration train throughput values (this is the throughput of the iteration with the
                lowest throughput).  None if unknown or unmeasured.
        """
        attributes = cls()
        assert_that(train_iterations_start is not None, "train_iterations_start is required.")
        assert_that(current_iteration is not None, "current_iteration is required.")
        assert_that(num_iterations is not None, "num_iterations is required.")
        assert_that(train_samples_start is not None, "train_samples_start is required.")
        assert_that(num_train_samples is not None, "num_train_samples is required.")
        assert_that(interval is not None, "interval is required.")
        assert_that(avg_iteration_time_sec is not None, "avg_iteration_time_sec is required.")
        assert_that(min_iteration_time_sec is not None, "min_iteration_time_sec is required.")
        assert_that(max_iteration_time_sec is not None, "max_iteration_time_sec is required.")
        assert_that(total_iteration_time_sec is not None, "total_iteration_time_sec is required.")
        attributes.add("train_iterations_start", train_iterations_start)
        attributes.add("current_iteration", current_iteration)
        attributes.add("num_iterations", num_iterations)
        attributes.add("train_samples_start", train_samples_start)
        attributes.add("num_train_samples", num_train_samples)
        attributes.add("interval", interval)
        attributes.add("avg_iteration_time_sec", avg_iteration_time_sec)
        attributes.add("min_iteration_time_sec", min_iteration_time_sec)
        attributes.add("max_iteration_time_sec", max_iteration_time_sec)
        attributes.add("total_iteration_time_sec", total_iteration_time_sec)
        if avg_forward_time_sec is not None:
            attributes.add("avg_forward_time_sec", avg_forward_time_sec)
        if avg_backward_time_sec is not None:
            attributes.add("avg_backward_time_sec", avg_backward_time_sec)
        if avg_dataloader_time_sec is not None:
            attributes.add("avg_dataloader_time_sec", avg_dataloader_time_sec)
        if avg_tflops is not None:
            attributes.add("avg_tflops", avg_tflops)
        if train_tokens is not None:
            attributes.add("train_tokens", train_tokens)
        if avg_tokens_per_second is not None:
            attributes.add("avg_tokens_per_second", avg_tokens_per_second)
        if latest_loss is not None:
            attributes.add("latest_loss", latest_loss)
        if avg_batch_size is not None:
            attributes.add("avg_batch_size", avg_batch_size)
        if completed_floating_point_operations_overall is not None:
            attributes.add("completed_floating_point_operations_overall", completed_floating_point_operations_overall)
        if total_flops is not None:
            attributes.add("total_flops", total_flops)
        if train_throughput_per_gpu is not None:
            attributes.add("train_throughput_per_gpu", train_throughput_per_gpu)
        if train_throughput_per_gpu_max is not None:
            attributes.add("train_throughput_per_gpu_max", train_throughput_per_gpu_max)
        if train_throughput_per_gpu_min is not None:
            attributes.add("train_throughput_per_gpu_min", train_throughput_per_gpu_min)
        return attributes

    @property
    def train_iterations_start(self) -> int:
        """The starting iteration number / global step(could be non-zero if the job loads a checkpoint and starts from there)."""
        val = self.get_int_value("train_iterations_start")
        assert_that(val is not None, "train_iterations_start is required.")
        return val  # type: ignore

    @property
    def current_iteration(self) -> int:
        """The current iteration number at the time the event is generated (includes the iterations from the loaded checkpoint, if any)."""
        val = self.get_int_value("current_iteration")
        assert_that(val is not None, "current_iteration is required.")
        return val  # type: ignore

    @property
    def num_iterations(self) -> int:
        """The number of iterations whose metrics are aggregated in this event.

        This is the same as the number of iterations performed in the current job. All the aggregated metrics
        below (avg_xxx, max_xxx, min_xxx, total_xxx) are computed over these iterations.
        """
        val = self.get_int_value("num_iterations")
        assert_that(val is not None, "num_iterations is required.")
        return val  # type: ignore

    @property
    def train_samples_start(self) -> int:
        """The starting sample number.

        Could be non-zero if the job loads a checkpoint and starts from there. This corresponds to the
        "train_iterations_start" attribute.
        """
        val = self.get_int_value("train_samples_start")
        assert_that(val is not None, "train_samples_start is required.")
        return val  # type: ignore

    @property
    def num_train_samples(self) -> int:
        """The number of samples processed during the "num_iterations".

        This is the same as the number of samples used for training in the current job.
        """
        val = self.get_int_value("num_train_samples")
        assert_that(val is not None, "num_train_samples is required.")
        return val  # type: ignore

    @property
    def interval(self) -> int:
        """Return the interval between the current and previous iteration.

        The interval represents the number of training iterations performed between
        the last TRAINING_MULTI_ITERATION_METRICS_UPDATE event and this one.

        Returns:
            int: The interval value.
        """
        val = self.get_int_value("interval")
        assert_that(val is not None, "interval is required.")
        return val  # type: ignore

    @property
    def avg_iteration_time_sec(self) -> float:
        """The average iteration time for the current job in seconds. None if unknown or unmeasured."""
        val = self.get_float_value("avg_iteration_time_sec")
        assert_that(val is not None, "avg_iteration_time_sec is required.")
        return val  # type: ignore

    @property
    def min_iteration_time_sec(self) -> float:
        """The minimum iteration time for the current job in seconds. None if unknown or unmeasured."""
        val = self.get_float_value("min_iteration_time_sec")
        assert_that(val is not None, "min_iteration_time_sec is required.")
        return val  # type: ignore

    @property
    def max_iteration_time_sec(self) -> float:
        """The maximum iteration time for the current job in seconds. None if unknown or unmeasured."""
        val = self.get_float_value("max_iteration_time_sec")
        assert_that(val is not None, "max_iteration_time_sec is required.")
        return val  # type: ignore

    @property
    def total_iteration_time_sec(self) -> float:
        """The total iteration time for the current job in seconds. None if unknown or unmeasured."""
        val = self.get_float_value("total_iteration_time_sec")
        assert_that(val is not None, "total_iteration_time_sec is required.")
        return val  # type: ignore

    @property
    def avg_forward_time_sec(self) -> Optional[float]:
        """The average forward time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("avg_forward_time_sec")

    @property
    def avg_backward_time_sec(self) -> Optional[float]:
        """The average backward time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("avg_backward_time_sec")

    @property
    def avg_dataloader_time_sec(self) -> Optional[float]:
        """The average dataloader time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("avg_dataloader_time_sec")

    @property
    def avg_tflops(self) -> Optional[float]:
        """The avgerage tflops (teraflops) for the current job."""
        return self.get_float_value("avg_tflops")

    @property
    def train_tokens(self) -> Optional[int]:
        """Number of training tokens processed so far in the current job. None if unknown or unmeasured."""
        return self.get_int_value("train_tokens")

    @property
    def avg_tokens_per_second(self) -> Optional[float]:
        """The tokens per second for the current job. None if unknown or unmeasured."""
        return self.get_float_value("avg_tokens_per_second")

    @property
    def latest_loss(self) -> Optional[float]:
        """The latest_loss. None if unknown or unmeasured."""
        return self.get_float_value("latest_loss")

    @property
    def avg_batch_size(self) -> Optional[int]:
        """The average batch size for the current job. None if unknown or unmeasured."""
        return self.get_int_value("avg_batch_size")

    @property
    def completed_floating_point_operations_overall(self) -> Optional[int]:
        """Number of floating point operations completed so far.

        Includes the ones from the loaded checkpoint and the ones from the current job.
        None if unknown or unmeasured.
        """
        return self.get_int_value("completed_floating_point_operations_overall")

    @property
    def total_flops(self) -> Optional[int]:
        """Total number of floating point operations in the current job. None if unknown or unmeasured."""
        return self.get_int_value("total_flops")

    @property
    def train_throughput_per_gpu(self) -> Optional[float]:
        """Train throughput per GPU during the current job in tflops (One trillion floating point operations per second).

        This is the average over the job so far. None if unknown or unmeasured.
        """
        return self.get_float_value("train_throughput_per_gpu")

    @property
    def train_throughput_per_gpu_max(self) -> Optional[float]:
        """The max train throughput per GPU during the current job in tflops (One trillion floating point operations per second).

        This value is computed as the max of the per-iteration train throughput values (this is the throughput of the iteration with the highest throughput).
        None if unknown or unmeasured.
        """
        return self.get_float_value("train_throughput_per_gpu_max")

    @property
    def train_throughput_per_gpu_min(self) -> Optional[float]:
        """The min train throughput per GPU during the current job in tflops (One trillion floating point operations per second).

        This value is computed as the min of the per-iteration train throughput values (this is the throughput of the iteration with the lowest throughput).
        None if unknown or unmeasured.
        """
        return self.get_float_value("train_throughput_per_gpu_min")


class ValidationMetricsUpdateAttributes(Attributes):
    """Event attributes for a VALIDATION_METRICS_UPDATE event of the VALIDATION_LOOP span for training jobs."""

    @classmethod
    def create(
        cls,
        current_iteration: int,
        interval: int,
        avg_iteration_time_sec: Optional[float] = None,
        min_iteration_time_sec: Optional[float] = None,
        max_iteration_time_sec: Optional[float] = None,
        total_iteration_time_sec: Optional[float] = None,
    ) -> "ValidationMetricsUpdateAttributes":
        """Create a ValidationMetricsUpdateAttributes object.

        Args:
            current_iteration: The current iteration number at the time the event is generated
                (includes the iterations from the loaded checkpoint, if any).
            interval: The interval between the current and previous iteration, where a similar event was reported. That is, the number of training iterations
                      performed between the last VALIDATION_METRICS_UPDATE event and this one.
            avg_iteration_time_sec: The average iteration time for the current job in seconds. None if unknown or unmeasured.
            min_iteration_time_sec: The minimum iteration time for the current job in seconds. None if unknown or unmeasured.
            max_iteration_time_sec: The maximum iteration time for the current job in seconds. None if unknown or unmeasured.
            total_iteration_time_sec: The total iteration time for the current job in seconds. None if unknown or unmeasured.
        """
        attributes = cls()
        assert_that(current_iteration is not None, "current_iteration is required.")
        assert_that(interval is not None, "interval is required.")
        attributes.add("current_iteration", current_iteration)
        attributes.add("interval", interval)
        if avg_iteration_time_sec is not None:
            attributes.add("avg_iteration_time_sec", avg_iteration_time_sec)
        if min_iteration_time_sec is not None:
            attributes.add("min_iteration_time_sec", min_iteration_time_sec)
        if max_iteration_time_sec is not None:
            attributes.add("max_iteration_time_sec", max_iteration_time_sec)
        if total_iteration_time_sec is not None:
            attributes.add("total_iteration_time_sec", total_iteration_time_sec)
        return attributes

    @property
    def current_iteration(self) -> int:
        """The current iteration number at the time the event is generated (includes the iterations from the loaded checkpoint, if any)."""
        val = self.get_int_value("current_iteration")
        assert_that(val is not None, "current_iteration is required.")
        return val  # type: ignore

    @property
    def interval(self) -> int:
        """Return the interval between the current and previous iteration.

        The interval represents the number of training iterations performed between
        the last VALIDATION_METRICS_UPDATE event and this one.

        Returns:
            int: The interval value.
        """
        val = self.get_int_value("interval")
        assert_that(val is not None, "interval is required.")
        return val  # type: ignore

    @property
    def avg_iteration_time_sec(self) -> Optional[float]:
        """The average iteration time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("avg_iteration_time_sec")

    @property
    def min_iteration_time_sec(self) -> Optional[float]:
        """The minimum iteration time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("min_iteration_time_sec")

    @property
    def max_iteration_time_sec(self) -> Optional[float]:
        """The maximum iteration time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("max_iteration_time_sec")

    @property
    def total_iteration_time_sec(self) -> Optional[float]:
        """The total iteration time for the current job in seconds. None if unknown or unmeasured."""
        return self.get_float_value("total_iteration_time_sec")


# Event attributes for a TESTING_METRICS_UPDATE event of the TESTING_LOOP span for training jobs.
# Currently, the attributes for the TESTING_METRICS_UPDATE event are the same as the attributes for the VALIDATION_METRICS_UPDATE event.
class TestingMetricsUpdateAttributes(Attributes):
    """Event attributes for a TESTING_METRICS_UPDATE event of the TESTING_LOOP span for training jobs."""

    @classmethod
    def create(
        cls,
        current_iteration: int,
        interval: int,
    ) -> "TestingMetricsUpdateAttributes":
        """Create a TestingMetricsUpdateAttributes object.

        Args:
            current_iteration: The current iteration number at the time the event is generated
                (includes the iterations from the loaded checkpoint, if any).
            interval: The interval between the current and previous iteration, where a similar event was reported. That is, the number of training iterations
                      performed between the last VALIDATION_METRICS_UPDATE event and this one.
            total_iteration_time_sec: The total iteration time for the current job in seconds. None if unknown or unmeasured.
        """
        attributes = cls()
        assert_that(current_iteration is not None, "current_iteration is required.")
        assert_that(interval is not None, "interval is required.")
        attributes.add("current_iteration", current_iteration)
        attributes.add("interval", interval)
        return attributes

    @property
    def current_iteration(self) -> int:
        """The current iteration number at the time the event is generated (includes the iterations from the loaded checkpoint, if any)."""
        val = self.get_int_value("current_iteration")
        assert_that(val is not None, "current_iteration is required.")
        return val  # type: ignore

    @property
    def interval(self) -> int:
        """The interval between the current and previous iteration, where a similar event was reported.

        That is, the number of training iterations performed between the last VALIDATION_METRICS_UPDATE event and this one.
        """
        val = self.get_int_value("interval")
        assert_that(val is not None, "interval is required.")
        return val  # type: ignore


class SaveCheckpointSuccessEventAttributes(Attributes):
    """Eventt atributes for various events related to saving checkpoints.

    These attributes are used for SAVE_CHECKPOINT_SUCCESS events for the spans of type
    CHECKPOINT_SAVE_SYNC or CHECKPOINT_SAVE_FINALIZATION for training jobs.
    """

    @classmethod
    def create(
        cls,
        checkpoint_strategy: CheckPointStrategy,
        current_iteration: int,
        first_successful_save_checkpoint_timestamp_sec: float,
        latest_successful_save_checkpoint_timestamp_sec: float,
        save_checkpoint_success_count: int,
        checkpoint_size: Optional[int] = None,
        checkpoint_directory: Optional[str] = None,
        training_start_timestamp_sec: Optional[float] = None,
    ) -> "SaveCheckpointSuccessEventAttributes":
        """Create a SaveCheckpointSuccessEventAttributes object.

        Args:
            checkpoint_strategy: The strategy used for saving checkpoints (SYNC or ASYNC).
            current_iteration: The current iteration number as of the time the checkpont save started.
            first_successful_save_checkpoint_timestamp_sec: The timestamp of the first successful save checkpoint.
                The timestamp represents the time the checkpont save operation finished as fractional seconds since epoch.
            latest_successful_save_checkpoint_timestamp_sec: The timestamp of the latest successful save checkpoint.
                The timestamp represents the time the checkpont save operation finished as fractional seconds since epoch.
            save_checkpoint_success_count: The number of times a checkpoint was succsffuly saved so far (in the current job).
            checkpoint_size: The size of the checkpoint in bytes.
            checkpoint_directory: The directory where the checkpoint is saved.
            training_start_timestamp_sec: The timestamp of the start of the training loop corresponding to this checkpoint.
                This shows when training started (and you can compare it with "first_successful_save_checkpoint_timestamp_sec"
                to see if we save checkpoints early enough to not lose too much work in case of failures).
                The timestamp represents fractional seconds since epoch.
                None if unknown or unmeasured.
        """
        attributes = cls()
        assert_that(checkpoint_strategy is not None, "checkpoint_strategy is required.")
        assert_that(current_iteration is not None, "current_iteration is required.")
        assert_that(first_successful_save_checkpoint_timestamp_sec is not None, "first_successful_save_checkpoint_timestamp_sec is required.")
        assert_that(latest_successful_save_checkpoint_timestamp_sec is not None, "latest_successful_save_checkpoint_timestamp_sec is required.")
        assert_that(save_checkpoint_success_count is not None, "save_checkpoint_success_count is required.")
        attributes.add("checkpoint_strategy", checkpoint_strategy)
        attributes.add("current_iteration", current_iteration)
        attributes.add("first_successful_save_checkpoint_timestamp_sec", first_successful_save_checkpoint_timestamp_sec)
        attributes.add("latest_successful_save_checkpoint_timestamp_sec", latest_successful_save_checkpoint_timestamp_sec)
        attributes.add("save_checkpoint_success_count", save_checkpoint_success_count)
        if checkpoint_size is not None:
            attributes.add("checkpoint_size", checkpoint_size)
        if checkpoint_directory is not None:
            attributes.add("checkpoint_directory", checkpoint_directory)
        if training_start_timestamp_sec:
            attributes.add("training_start_timestamp_sec", training_start_timestamp_sec)

        return attributes

    @property
    def checkpoint_strategy(self) -> CheckPointStrategy:
        """The strategy used for saving checkpoints (SYNC or ASYNC)."""
        val = self.get_str_value("checkpoint_strategy")
        assert_that(val is not None, "checkpoint_strategy is required.")
        return CheckPointStrategy(val)

    @property
    def current_iteration(self) -> int:
        """The current iteration number as of the time the checkpont save started."""
        val = self.get_int_value("current_iteration")
        assert_that(val is not None, "current_iteration is required.")
        return val  # type: ignore

    @property
    def first_successful_save_checkpoint_timestamp_sec(self) -> float:
        """The timestamp of the first successful save checkpoint.

        The timestamp represents the time the checkpont save operation finished as fractional seconds since epoch.
        """
        val = self.get_float_value("first_successful_save_checkpoint_timestamp_sec")
        assert_that(val is not None, "first_successful_save_checkpoint_timestamp_sec is required.")
        return val  # type: ignore

    @property
    def latest_successful_save_checkpoint_timestamp_sec(self) -> float:
        """The timestamp of the latest successful save checkpoint.

        The timestamp represents the time the checkpont save operation finished as fractional seconds since epoch.
        """
        val = self.get_float_value("latest_successful_save_checkpoint_timestamp_sec")
        assert_that(val is not None, "latest_successful_save_checkpoint_timestamp_sec is required.")
        return val  # type: ignore

    @property
    def save_checkpoint_success_count(self) -> int:
        """The number of times a checkpoint was succsffuly saved so far (in the current job)."""
        val = self.get_int_value("save_checkpoint_success_count")
        assert_that(val is not None, "save_checkpoint_success_count is required.")
        return val  # type: ignore

    @property
    def checkpoint_size(self) -> Optional[int]:
        """The size of the checkpoint in bytes."""
        return self.get_int_value("checkpoint_size")

    @property
    def checkpoint_directory(self) -> Optional[str]:
        """The directory where the checkpoint is saved."""
        return self.get_str_value("checkpoint_directory")

    @property
    def training_start_timestamp_sec(self) -> Optional[float]:
        """The timestamp of the start of the training loop corresponding to this checkpoint.

        This shows when training started (and you can compare it with "first_successful_save_checkpoint_timestamp_sec"
        to see if we save checkpoints early enough to not lose too much work in case of failures).
        The timestamp represents fractional seconds since epoch.

        None if unknown or unmeasured.
        """
        return self.get_float_value("training_start_timestamp_sec")


class SyncCheckpointMetricsUpdateAttributes(Attributes):
    """Event attributes for a SYNC_CHECKPOINT_METRICS_UPDATE event of the CHECKPOINT_SAVE_SYNC span for training jobs."""

    @classmethod
    def create(
        cls, save_checkpoint_sync_time_total_sec: float, save_checkpoint_sync_time_min_sec: float, save_checkpoint_sync_time_max_sec: float
    ) -> "SyncCheckpointMetricsUpdateAttributes":
        """Create a SyncCheckpointMetricsUpdateAttributes object.

        Args:
            save_checkpoint_sync_time_total_sec: The total time taken for all the sync checkpoint save operations
                in the current job in seconds.
            save_checkpoint_sync_time_min_sec: The minimum time taken for any sync checkpoint save operation in seconds.
            save_checkpoint_sync_time_max_sec: The maximum time taken for any sync checkpoint save operation in seconds.
        """
        attributes = cls()
        assert_that(save_checkpoint_sync_time_total_sec is not None, "save_checkpoint_sync_time_total_sec is required.")
        assert_that(save_checkpoint_sync_time_min_sec is not None, "save_checkpoint_sync_time_min_sec is required.")
        assert_that(save_checkpoint_sync_time_max_sec is not None, "save_checkpoint_sync_time_max_sec is required.")
        attributes.add("save_checkpoint_sync_time_total_sec", save_checkpoint_sync_time_total_sec)
        attributes.add("save_checkpoint_sync_time_min_sec", save_checkpoint_sync_time_min_sec)
        attributes.add("save_checkpoint_sync_time_max_sec", save_checkpoint_sync_time_max_sec)
        return attributes

    @property
    def save_checkpoint_sync_time_total_sec(self) -> float:
        """The total time taken for all the sync checkpoint save operations in the current job in seconds."""
        val = self.get_float_value("save_checkpoint_sync_time_total_sec")
        assert_that(val is not None, "save_checkpoint_sync_time_total_sec is required.")
        return val  # type: ignore

    @property
    def save_checkpoint_sync_time_min_sec(self) -> float:
        """The minimum time taken for any sync checkpoint save operation in seconds."""
        val = self.get_float_value("save_checkpoint_sync_time_min_sec")
        assert_that(val is not None, "save_checkpoint_sync_time_min_sec is required.")
        return val  # type: ignore

    @property
    def save_checkpoint_sync_time_max_sec(self) -> float:
        """The maximum time taken for any sync checkpoint save operation in seconds."""
        val = self.get_float_value("save_checkpoint_sync_time_max_sec")
        assert_that(val is not None, "save_checkpoint_sync_time_max_sec is required.")
        return val  # type: ignore
