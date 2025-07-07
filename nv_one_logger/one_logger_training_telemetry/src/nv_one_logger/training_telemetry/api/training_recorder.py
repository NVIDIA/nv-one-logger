# SPDX-License-Identifier: Apache-2.0
"""
This file contains the TrainingRecorder class, which is responsible for recording training telemetry data.

The TrainingRecorder class extends the DefaultRecorder class and provides specialized recording capabilities
for training-related telemetry data.

"""
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.metric_summarizer import MetricSummarizer
from nv_one_logger.core.internal.multi_window_timer import MultiWindowTimer
from nv_one_logger.core.internal.version import get_version
from nv_one_logger.core.span import Span, SpanName, StandardSpanName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.recorder.default_recorder import DefaultRecorder, ExportCustomizationMode
from overrides import override  # type: ignore[ancereportUnknownVariableType]

from nv_one_logger.training_telemetry.api.attributes import (
    CheckpointSaveSpanAttributes,
    OneLoggerInitializationAttributes,
    SaveCheckpointSuccessEventAttributes,
    SyncCheckpointMetricsUpdateAttributes,
    TestingMetricsUpdateAttributes,
    TrainingLoopAttributes,
    TrainingMetricsUpdateAttributes,
    ValidationMetricsUpdateAttributes,
)
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _create_multi_iteration_timers() -> Dict[StandardTrainingJobSpanName, MultiWindowTimer]:
    """Create a dictionary of multi-iteration timers.

    Returns:
        Dict[SpanName, MultiWindowTimer]: A dictionary mapping span names to their timers.
    """
    return {
        # A timer that keep track of training windows (training iterations).
        StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION: MultiWindowTimer(),
        # A timer that keep track of validation windows (validation iterations).
        StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION: MultiWindowTimer(),
        # A timer that keep track of synchronoussave checkpoint windows (all save checkpoint operations).
        StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC: MultiWindowTimer(),
        # A timer that keep track of synchronoussave checkpoint windows (all save checkpoint operations).
        StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC: MultiWindowTimer(),
        # A timer that keep track of load checkpoint windows (all load checkpoint operations).
        StandardTrainingJobSpanName.CHECKPOINT_LOAD: MultiWindowTimer(),
    }


@dataclass
class _TrainingState:
    """Internal state for tracking training progress and metrics.

    This class maintains state about the training process, including:
    - FLOPS calculations
    - Training iterations
    - Multi-iteration timers
    - Training samples
    - Various timestamps
    """

    # A dictionary that keeps track of the timers for each operations that is done in multiple iterations.
    # This is needed because operations such as training a batch, validation using a batch, and checkpoint save/load
    # are done multiple times in a single job (possibly with some time in between doing something else). We want to
    # aggregate some metrics over multiple iterations (e.g., report the total time spent on saving checkpoints OR
    # report aggregate training iterations metrics over N consecutive training iterations). So we use a
    # MultiWindowTimer for any operation that needs aggregatation over multiple iterations.
    # How often we reset the timer depends on the operation. For example, we reset the timer for training iterations
    # every N iterations where N == config.log_every_n_train_iterations  but for checkpoint saving, we aggregate
    # over all checkpoint save operations and never reset the timer.
    multi_iteration_timers: Dict[StandardTrainingJobSpanName, MultiWindowTimer] = field(default_factory=_create_multi_iteration_timers)

    # The starting iteration number (could be non-zero if the job loads a checkpoint and starts from there).
    train_iterations_start: int = 0

    # Completed training iterations (including iterations from the loaded checkpoint, i.e.,
    # iterations before "train_iterations_start").
    completed_training_iterations_overall: int = 0

    # The starting sample number (could be non-zero if the job loads a checkpoint and starts from there).
    # This corresponds to the "train_iterations_start" attribute.
    train_samples_start: int = 0

    # Number of training samples processed so far in the current job (does NOT include the samples from the loaded checkpoint).
    train_samples_start_processed_current_job: int = 0

    # Total number of floating point operations in the current job.
    total_flops_current_job: int = 0

    # Number of training tokens processed so far in the current job or None if the sequence length is not known.
    # None if unknown or unmeasured.
    train_tokens_current_job: Optional[int] = None

    # Number of floating point operations completed so far (including the ones from the loaded checkpoint and
    # the ones from the current job). None if unknown or unmeasured.
    completed_floating_point_operations_overall: Optional[int] = None

    # The timestamp of the end of the first training loop that was logged.
    first_logged_train_iterations_finish_time: Optional[TracingTimestamp] = None

    # The timestamp of the end of the latest training loop that was logged.
    last_logged_train_iterations_finish_time: Optional[TracingTimestamp] = None

    # The timestamp of the first successful save checkpoint.
    first_save_checkpoint_success_time: Optional[TracingTimestamp] = None

    # The timestamp of the latest successful save checkpoint.
    latest_save_checkpoint_success_time: Optional[TracingTimestamp] = None

    # Keeps track of the value of "completed_training_iterations_overall" at the time we performed the latest validation loop.
    # Initially, set to "completed_training_iterations_overall".
    validation_interval_start: int = 0

    # Keeps track of the value of "completed_training_iterations_overall" at the time we performed the latest testing loop.
    # Initially, set to "completed_training_iterations_overall".
    testing_interval_start: int = 0

    # A metric summarizer for tracking the train throughput per GPU in tflops (one trillion floating point operations per second).
    tflops_per_gpu: MetricSummarizer[float] = field(default_factory=lambda: MetricSummarizer[float]())

    # Number of checkpoints successfully saved in the current job.
    successful_save_checkpoint_count_current_job: int = 0


class TrainingRecorder(DefaultRecorder):
    """A recorder specifically designed for training telemetry.

    This class extends DefaultRecorder to provide specialized recording capabilities
    for training-related telemetry data.
    """

    def __init__(
        self,
        config: TrainingTelemetryConfig,
        exporters: List[Exporter],
        export_customization_mode: ExportCustomizationMode,
        span_name_filter: Optional[List[SpanName]],
    ):
        """Initialize the TrainingRecorder with a list of exporters.

        Args:
            config: The configuration for the training telemetry.
            exporters: A list of exporters to use for recording training telemetry.
            export_customization_mode: The mode of exporting spans (and their associated events and attribytes) to exporters.
            span_name_filter: This argument should be interpretted wrt the value of export_customization_mode:
                If export_customization_mode is ExportCustomizationMode.EXPORT_ALL_SPANS, span_name_filter should not be set.
                If export_customization_mode is ExportCustomizationMode.WHITELIST_SPANS, span_name_filter is a list of span names to export (whitelist).
                If export_customization_mode is ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter is a list of span names to not export (blacklist).
        """
        super().__init__(exporters, export_customization_mode=export_customization_mode, span_name_filter=span_name_filter)

        self._config: TrainingTelemetryConfig = config

        self._training_state = _TrainingState()
        if config.is_log_throughput_enabled:
            self._training_state.completed_floating_point_operations_overall = 0

    def _get_active_span(self, span_name: Union[StandardTrainingJobSpanName, StandardSpanName]) -> Span:
        """Return a single active span with the given name.

        This helper function is meant to be used when the caller knows that there is exactly one active span with the given name.

        Unlike using timed_span or context managers, when using callbacks for training telemetry,
        the callback function that creates a span and the callback function that stops the span
        are separate and there is no way to pass the span object to the callback function that stops the span.
        So for callbacks that need to stop a span, we need another way to find that span:
        The recorder start() and stop() methods ensure that for all standard (predefined) training spans, we
        do not allow callbacks to create a new span if a span with the same name is already active.
        This then allows us to find standard training spans by name unambiguously because at any given point in time,
        we have at most one active span with any given name.
        """
        assert_that(
            span_name in StandardTrainingJobSpanName or span_name in StandardSpanName,
            f"This function works only for standard (predefined) training spans.Invalid span name: {span_name}",
        )
        spans = self.get_active_spans_by_name(span_name)
        assert_that(len(spans) == 1, f"Expected to have one and only one span named {span_name} but found {len(spans)}.")
        return spans[0]

    @override
    def start(
        self,
        span_name: SpanName,
        span_attributes: Optional[Attributes] = None,
        start_event_attributes: Optional[Attributes] = None,
        start_time: Optional[TracingTimestamp] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a new training span.

        Args:
            span_name: The name of the span to start.
            span_attributes: Optional attributes to attach to the span.
            start_event_attributes: Optional attributes to attach to the start event.
            start_time: Optional timestamp for when the span started.
            parent_span: Optional The parent span of the new span. If not specified, the new span will be created as a child of the latest active span
            (or will be a root span if there is no active span).

        Returns:
            Span: The newly created span.
        """
        if not start_time:
            start_time = TracingTimestamp.now()

        # For standard (predefined) training spans, we don't allow two active spans with the same name. See the comments on _get_active_span.
        if span_name in StandardTrainingJobSpanName or span_name in StandardSpanName:
            spans = self.get_active_spans_by_name(span_name)
            if len(spans) > 0:
                raise OneLoggerError(
                    f"Cannot start span {span_name} while {len(spans)} span(s) with the same name is already active."
                    " Please ensure the callback is called correctly."
                )

        # Extra check to make sure the timer has been started (if this spans corresponds to a multi-iteration operation).
        # You may be tempted to start the timer here but this wouldn't work because for some of the on_xxx_start methods,
        # we first need to start the timers, then use the updated timer stats to set the attributes for spans or events,
        # and then we need to call super().start() to start the span. So we are leaving the responsibility of starting/stopping the timers to
        # the individual on_xxx_start methods but do a check here to catch cases that the timer is not started there.
        if span_name in self._training_state.multi_iteration_timers.keys():
            assert_that(
                self._training_state.multi_iteration_timers[span_name].is_active,  # type: ignore[reportArgumentType]
                f"Timer for span {span_name} is not active.",
            )

        return super().start(
            span_name=span_name, span_attributes=span_attributes, start_event_attributes=start_event_attributes, start_time=start_time, parent_span=parent_span
        )

    @override
    def stop(
        self,
        span: Span,
        stop_event_attributes: Optional[Attributes] = None,
        stop_time: Optional[TracingTimestamp] = None,
    ) -> None:
        """Stop a training span.

        Args:
            span: The span to stop.
            stop_event_attributes: Optional attributes to attach to the stop event.
            stop_time: Optional timestamp for when the span stopped.
        """
        if not stop_time:
            stop_time = TracingTimestamp.now()

        # Extra check to make sure the timer has been stopped (if this spans corresponds to a multi-iteration operation).
        # You may be tempted to stop the timer here but this wouldn't work because for some of the on_xxx_end methods,
        # we first need to stop the timers, then use the updated timer stats to set the attributes for spans or events,
        # and then we need to call super().stop() to stop the span. So we are leaving the responsibility of stopping the timers to
        # the individual on_xxx_stop methods but do a check here to catch cases that the timer is not stopped there.
        if span.name in self._training_state.multi_iteration_timers.keys():
            assert_that(
                not self._training_state.multi_iteration_timers[span.name].is_active,  # type: ignore[reportArgumentType]
                f"Timer for span {span.name} is still active.",
            )

        super().stop(
            span=span,
            stop_event_attributes=stop_event_attributes,
            stop_time=stop_time,
        )

    @override
    def event(self, span: Span, event: Event) -> Event:
        """Add an event to a training span.

        Args:
            span: The span to add the event to.
            event: The event to add.

        Returns:
            Event: The added event.
        """
        event = super().event(span=span, event=event)
        return event

    def on_app_start(self, start_time: TracingTimestamp) -> Span:
        """Start the application span, update state if necessary, and then add the one logger initialization event.

        Args:
            start_time: The timestamp of the start of the application.

        Returns:
            Span: The newly created span for the application.
        """
        app_span = self.start(
            span_name=StandardSpanName.APPLICATION,
            start_time=start_time,
        )

        conf = self._config
        attributes = OneLoggerInitializationAttributes.create(
            one_logger_training_telemetry_version=get_version("one-logger-training-telemetry"),
            enable_for_current_rank=conf.enable_for_current_rank,
            perf_tag=conf.perf_tag,
            session_tag=conf.session_tag,
            app_type=conf.app_type,
            log_every_n_train_iterations=conf.log_every_n_train_iterations,
            world_size=conf.world_size,
            global_batch_size=conf.global_batch_size,
            is_baseline_run=conf.is_baseline_run,
            is_train_iterations_enabled=conf.is_train_iterations_enabled,
            is_validation_iterations_enabled=conf.is_validation_iterations_enabled,
            is_test_iterations_enabled=conf.is_test_iterations_enabled,
            is_save_checkpoint_enabled=conf.is_save_checkpoint_enabled,
            is_log_throughput_enabled=conf.is_log_throughput_enabled,
            summary_data_schema_version=conf.summary_data_schema_version,
            rank=_get_rank(),
            checkpoint_strategy=conf.save_checkpoint_strategy,
            micro_batch_size=conf.micro_batch_size,
            seq_length=conf.seq_length,
            custom_metadata=conf.custom_metadata,
            node_name=socket.gethostname(),
        )
        self.event(
            app_span,
            Event.create(
                name=StandardTrainingJobEventName.ONE_LOGGER_INITIALIZATION,
                attributes=attributes,
                timestamp=start_time,
            ),
        )
        return app_span

    def on_app_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the application span, update state if necessary, and then close the recorder.

        Args:
            stop_time: The timestamp of the end of the application.
        """
        self.stop(
            span=self._get_active_span(StandardSpanName.APPLICATION),
            stop_time=stop_time,
        )

        # Finalize everything and clean up.
        self.close()

    def on_model_init_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for model initialization, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of model initialization.

        Returns:
            Span: The newly created span for model initialization.
        """
        return self.start(
            span_name=StandardTrainingJobSpanName.MODEL_INIT,
            start_time=start_time,
        )

    def on_model_init_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the model initialization span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of model initialization.
        """
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.MODEL_INIT),
            stop_time=stop_time,
        )

    def on_dataloader_init_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for dataloader initialization.

        Args:
            start_time: The timestamp of the start of dataloader initialization.

        Returns:
            Span: The newly created span for dataloader initialization.
        """
        return self.start(
            span_name=StandardTrainingJobSpanName.DATA_LOADER_INIT,
            start_time=start_time,
        )

    def on_dataloader_init_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the dataloader initialization span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of dataloader initialization.
        """
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.DATA_LOADER_INIT),
            stop_time=stop_time,
        )

    def on_load_checkpoint_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for checkpoint loading, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of checkpoint loading.

        Returns:
            Span: The newly created span for checkpoint loading.
        """
        # Step 1: Update the state.
        self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_LOAD].start(start_time)

        # Step 2: Create the span.
        return self.start(
            span_name=StandardTrainingJobSpanName.CHECKPOINT_LOAD,
            start_time=start_time,
        )

    def on_load_checkpoint_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the checkpoint loading span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of checkpoint loading.
        """
        # Step 1: Update the state.
        self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_LOAD].stop(stop_time)

        # Step 2: Stop the span.
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_LOAD),
            stop_time=stop_time,
        )

    def on_optimizer_init_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for optimizer initialization, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of optimizer initialization.

        Returns:
            Span: The newly created span for optimizer initialization.
        """
        return self.start(
            span_name=StandardTrainingJobSpanName.OPTIMIZER_INIT,
            start_time=start_time,
        )

    def on_optimizer_init_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the optimizer initialization span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of optimizer initialization.
        """
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.OPTIMIZER_INIT),
            stop_time=stop_time,
        )

    def on_training_loop_start(
        self,
        train_iterations_start: int,
        train_samples_start: int,
        train_iterations_target: Optional[int] = None,
        train_samples_target: Optional[int] = None,
        train_tokens_target: Optional[int] = None,
        start_time: Optional[TracingTimestamp] = None,
    ) -> Span:
        """Start a new span for training loop, and update the state if necessary.

        Args:
            train_iterations_start: The starting iteration number / global step(could be non-zero if the job loads a checkpoint and starts from there).
            train_samples_start: The starting sample number (could be non-zero if the job loads a checkpoint and starts from there).
            train_iterations_target: Target number of training iterations.
            train_samples_target: Target number of training samples.
            train_tokens_target: Target numbrer of training tokens.
            start_time: Optional timestamp for when the training loop started.

        Returns:
            TrainingLoopAttributes for a new StandardTrainingJobSpanName.TRAINING_LOOP span.
        """
        assert_that(
            train_iterations_start >= 0,
            f"Invalid value for train_iterations_start in TrainingLoopAttributes object: {train_iterations_start}",
        )
        assert_that(
            train_samples_start >= 0,
            f"Invalid value for train_samples_start in TrainingLoopAttributes object: {train_samples_start}",
        )

        # Step 1: Update the state.
        state = self._training_state
        state.train_iterations_start = train_iterations_start
        state.train_samples_start = train_samples_start
        # We assume the first iteration is iteration 0. So completed_training_iterations_overall is the same as train_iterations_start.
        state.completed_training_iterations_overall = train_iterations_start
        # iteration number (global step) is zero-based. So if completed_training_iterations_overall is N, the next training iteration will be iteration N.
        state.validation_interval_start = state.completed_training_iterations_overall
        state.testing_interval_start = state.completed_training_iterations_overall
        if self._config.is_log_throughput_enabled:
            assert_that(
                self._config.flops_per_sample and self._config.flops_per_sample > 0,
                "flops_per_sample must be set to a positive value when is_log_throughput_enabled is True",
            )
            # The initial value of completed_floating_point_operations_overall is nonzero if loading ckpt, whereas total_flops_current_job
            # is always initialized tozero. For example, if train_iterations_start is 1,  it means that one iteration (iteration 0) has
            # been completed in a previous run.
            state.completed_floating_point_operations_overall = (
                train_iterations_start * self._config.global_batch_size * self._config.flops_per_sample
            )  # type: ignore
            state.total_flops_current_job = 0

        # Step 2: Create the span.
        span_attributes = TrainingLoopAttributes.create(
            train_iterations_start=train_iterations_start,
            train_samples_start=train_samples_start,
            train_iterations_target=train_iterations_target,
            train_samples_target=train_samples_target,
            train_tokens_target=train_tokens_target,
            completed_floating_point_operations_overall=state.completed_floating_point_operations_overall,
        )
        return self.start(
            span_name=StandardTrainingJobSpanName.TRAINING_LOOP,
            span_attributes=span_attributes,
            start_time=start_time,
        )

    def on_training_loop_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the training loop span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of training loop.
        """
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.TRAINING_LOOP),
            stop_time=stop_time,
        )

    def on_training_single_iteration_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for a single training iteration, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of the training iteration.

        Returns:
            Span: The newly created span for the training iteration.
        """
        # Step 1: Update the state.
        self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION].start(start_time)

        # Step 2: Create the span.
        return self.start(
            span_name=StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION,
            start_time=start_time,
        )

    def on_training_single_iteration_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the training iteration span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of the training iteration.
        """
        # Step 1: Update the state.
        training_iteration_timer = self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        training_iteration_timer.stop(stop_time)

        self._training_state.completed_training_iterations_overall += 1
        self._training_state.train_samples_start_processed_current_job += self._config.global_batch_size
        if self._config.seq_length:
            self._training_state.train_tokens_current_job = self._config.seq_length * self._training_state.train_samples_start_processed_current_job

        self._training_state.last_logged_train_iterations_finish_time = stop_time
        if not self._training_state.first_logged_train_iterations_finish_time:
            self._training_state.first_logged_train_iterations_finish_time = stop_time

        if self._config.is_log_throughput_enabled:
            assert_that(
                self._config.flops_per_sample and self._config.flops_per_sample > 0,
                "flops_per_sample must be set to a positive value when is_log_throughput_enabled is True",
            )
            flops = self._config.global_batch_size * self._config.flops_per_sample  # type: ignore[reportOperatorIssue]
            assert_that(
                self._training_state.completed_floating_point_operations_overall is not None, "completed_floating_point_operations_overall must be initialized."
            )
            self._training_state.completed_floating_point_operations_overall += flops  # type: ignore[reportOperatorIssue]
            assert_that(
                self._training_state.total_flops_current_job is not None,  # type: ignore[reportUnnecesaryComparison]
                "Must be initialized! Did you start the TRAINING_LOOP span?",
            )
            self._training_state.total_flops_current_job += flops
            train_iterations_time_total = training_iteration_timer.total_time_sec
            assert_that(train_iterations_time_total > 0, "train_iterations_time_total must be greater than 0")
            assert_that(self._config.world_size > 0, "world_size must be greater than 0")
            train_throughput_per_gpu = float(self._training_state.total_flops_current_job) / (train_iterations_time_total * 10**12 * self._config.world_size)
            self._training_state.tflops_per_gpu.add_value(train_throughput_per_gpu)

        # Step 2: Send updated telemetry data on training every N train iterations.
        self._maybe_send_training_metrics_update()

        # Step 2: Stop the span.
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION),
            stop_time=stop_time,
        )

    def _maybe_send_training_metrics_update(self) -> None:
        """Send updated telemetry data on training every N train iterations."""
        # iteration number (global step) is zero-based. So if completed_training_iterations_overall is N,
        # the last completed training iteration was iteration N-1.
        latest_iteration = self._training_state.completed_training_iterations_overall - 1
        if latest_iteration > 0 and latest_iteration % self._config.log_every_n_train_iterations == 0:
            training_loop_span = self._get_active_span(StandardTrainingJobSpanName.TRAINING_LOOP)
            training_iteration_timer = self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
            attributes = TrainingMetricsUpdateAttributes.create(
                train_iterations_start=self._training_state.train_iterations_start,
                current_iteration=latest_iteration,
                num_iterations=training_iteration_timer.total_window_count,
                train_samples_start=self._training_state.train_samples_start,
                num_train_samples=self._training_state.train_samples_start_processed_current_job,
                interval=self._config.log_every_n_train_iterations,
                avg_iteration_time_sec=training_iteration_timer.avg_window_duration_sec,
                min_iteration_time_sec=training_iteration_timer.min_window_duration_sec,
                max_iteration_time_sec=training_iteration_timer.max_window_duration_sec,
                total_iteration_time_sec=training_iteration_timer.total_time_sec,
                train_tokens=self._training_state.train_tokens_current_job,
                completed_floating_point_operations_overall=self._training_state.completed_floating_point_operations_overall,
                total_flops=self._training_state.total_flops_current_job,
                train_throughput_per_gpu=self._training_state.tflops_per_gpu.latest_value,
                train_throughput_per_gpu_max=self._training_state.tflops_per_gpu.max_value,
                train_throughput_per_gpu_min=self._training_state.tflops_per_gpu.min_value,
                first_logged_train_iterations_finish_timestamp_sec=(
                    self._training_state.first_logged_train_iterations_finish_time.seconds_since_epoch
                    if self._training_state.first_logged_train_iterations_finish_time
                    else None
                ),
            )
            self.event(training_loop_span, Event.create(name=StandardTrainingJobEventName.TRAINING_METRICS_UPDATE, attributes=attributes))

    def on_validation_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for validation loop, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of validation.

        Returns:
            Span: The newly created span for validation.
        """
        return self.start(
            span_name=StandardTrainingJobSpanName.VALIDATION_LOOP,
            start_time=start_time,
        )

    def on_validation_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the validation loop span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of validation.
        """
        validation_loop_span = self._get_active_span(StandardTrainingJobSpanName.VALIDATION_LOOP)

        # Step 1: Update the state.
        complete_training_iters = self._training_state.completed_training_iterations_overall
        assert_that(
            self._training_state.validation_interval_start >= 0,
            f"Validation interval start invalid: {self._training_state.validation_interval_start}. complete_training_iters: {complete_training_iters}",
        )
        validation_iteration_timer = self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION]
        # This helps us deal with a case that we get callbacks for the validation loop but not individual validation iterations.
        # This is a likely scenario because unline training, for validation we send the metric update events only at the end of the validation loop.
        measured_validation_iterations = validation_iteration_timer.total_window_count

        # Step 2: Send updated telemetry data on validation.
        attributes = ValidationMetricsUpdateAttributes.create(
            # Iteration number (global step) is zero-based. So if completed_training_iterations_overall is N,
            # the last completed training iteration was iteration N-1.
            current_iteration=max(0, complete_training_iters - 1),
            interval=complete_training_iters - self._training_state.validation_interval_start,
            avg_iteration_time_sec=validation_iteration_timer.avg_window_duration_sec if measured_validation_iterations > 0 else None,
            min_iteration_time_sec=validation_iteration_timer.min_window_duration_sec if measured_validation_iterations > 0 else None,
            max_iteration_time_sec=validation_iteration_timer.max_window_duration_sec if measured_validation_iterations > 0 else None,
            total_iteration_time_sec=validation_iteration_timer.total_time_sec if measured_validation_iterations > 0 else None,
        )
        self.event(validation_loop_span, Event.create(name=StandardTrainingJobEventName.VALIDATION_METRICS_UPDATE, attributes=attributes))
        self._training_state.validation_interval_start = complete_training_iters

        # Step 3: Stop the span.
        self.stop(
            span=validation_loop_span,
            stop_time=stop_time,
        )

    def on_validation_single_iteration_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for a single validation iteration, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of the validation iteration.

        Returns:
            Span: The newly created span for the validation iteration.
        """
        # Step 1: Update the state.
        self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION].start(start_time)

        # Step 2: Create the span.
        return self.start(
            span_name=StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION,
            start_time=start_time,
        )

    def on_validation_single_iteration_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the validation iteration span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of the validation iteration.
        """
        # Step 1: Update the state.
        self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION].stop(stop_time)

        # Step 2: Stop the span.
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION),
            stop_time=stop_time,
        )

    def on_testing_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for testing loop, and update the state if necessary.

        Args:
            start_time: The timestamp of the start of testing.

        Returns:
            Span: The newly created span for testing.
        """
        return self.start(
            span_name=StandardTrainingJobSpanName.TESTING_LOOP,
            start_time=start_time,
        )

    def on_testing_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the testing loop span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of testing.
        """
        testing_loop_span = self._get_active_span(StandardTrainingJobSpanName.TESTING_LOOP)
        # Step 1: Update the state.
        complete_training_iters = self._training_state.completed_training_iterations_overall
        assert_that(
            self._training_state.testing_interval_start >= 0,
            f"Testing interval start invalid: {self._training_state.testing_interval_start}. current_iteration: {complete_training_iters}",
        )
        attributes = TestingMetricsUpdateAttributes.create(
            # Iteration number (global step) is zero-based. So if completed_training_iterations_overall is N,
            # the last completed training iteration was iteration N-1.
            current_iteration=max(0, complete_training_iters - 1),
            interval=complete_training_iters - self._training_state.testing_interval_start,
        )

        # Step 2: Send updated telemetry data on testing.
        self.event(testing_loop_span, Event.create(name=StandardTrainingJobEventName.TESTING_METRICS_UPDATE, attributes=attributes))
        self._training_state.testing_interval_start = complete_training_iters

        # Step 3: Stop the span.
        self.stop(
            span=testing_loop_span,
            stop_time=stop_time,
        )

    def create_sync_checkpoint_metrics_event(self) -> Event:
        """Create an event of type SYNC_CHECKPOINT_METRICS_UPDATE using the most recent checkpoint metrics."""
        sync_checkpoint_timer = self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC]
        attributes = SyncCheckpointMetricsUpdateAttributes.create(
            save_checkpoint_sync_time_total_sec=sync_checkpoint_timer.total_time_sec,
            save_checkpoint_sync_time_min_sec=sync_checkpoint_timer.min_window_duration_sec,
            save_checkpoint_sync_time_max_sec=sync_checkpoint_timer.max_window_duration_sec,
        )
        return Event.create(name=StandardTrainingJobEventName.SYNC_CHECKPOINT_METRICS_UPDATE, attributes=attributes)

    def on_save_checkpoint_start(self, current_iteration: int, start_time: TracingTimestamp) -> Span:
        """Start a new span for checkpoint saving, and update the state if necessary.

        Args:
            current_iteration: The current iteration number.
            start_time: The timestamp of the start of the checkpoint saving.

        Returns:
            Span: The newly created span for checkpoint saving.
        """
        # Step 1: Update the state.
        span_name = None
        if self._config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            span_name = StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
        elif self._config.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            span_name = StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC
        else:
            raise OneLoggerError(f"Invalid checkpoint strategy: {self._config.save_checkpoint_strategy}")

        timer = self._training_state.multi_iteration_timers[span_name]
        timer.start(start_time)

        # Step 2: Create the span.
        span_attributes = CheckpointSaveSpanAttributes.create(
            self._config.save_checkpoint_strategy,
            current_iteration=current_iteration,
            # The current save attempt is already included in the total window count.
            save_checkpoint_attempt_count=timer.total_window_count,
        )
        return self.start(span_name=span_name, span_attributes=span_attributes)

    def on_save_checkpoint_success(self, current_iteration: int, timestamp: TracingTimestamp) -> None:
        """Send an event of type SAVE_CHECKPOINT_SUCCESS and update the state if necessary.

        Args:
            current_iteration: The current iteration number.
            timestamp: The timestamp of the checkpoint saving.
        """
        # Step 1: Update the state.
        conf = self._config
        parent_span = None
        # See comments on StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS.
        if conf.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            parent_span = self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC)
        elif conf.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            parent_span = self._get_active_span(StandardTrainingJobSpanName.TRAINING_LOOP)
        else:
            raise OneLoggerError(f"Invalid checkpoint strategy: {conf.save_checkpoint_strategy}")

        state = self._training_state
        state.successful_save_checkpoint_count_current_job += 1
        state.latest_save_checkpoint_success_time = timestamp
        if not state.first_save_checkpoint_success_time:
            state.first_save_checkpoint_success_time = timestamp

        # Step 2: Create the event.
        event_attributes = SaveCheckpointSuccessEventAttributes.create(
            checkpoint_strategy=conf.save_checkpoint_strategy,
            current_iteration=current_iteration,
            first_successful_save_checkpoint_timestamp_sec=state.first_save_checkpoint_success_time.seconds_since_epoch,
            latest_successful_save_checkpoint_timestamp_sec=state.latest_save_checkpoint_success_time.seconds_since_epoch,
            save_checkpoint_success_count=state.successful_save_checkpoint_count_current_job,
            training_start_timestamp_sec=(
                state.first_logged_train_iterations_finish_time.seconds_since_epoch if state.first_logged_train_iterations_finish_time else None
            ),
        )
        self.event(
            parent_span,
            Event.create(name=StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS, attributes=event_attributes, timestamp=timestamp),
        )

    def on_save_checkpoint_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the save checkpoint span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of testing.
        """
        # Step 1: Update the state.
        span = None
        if self._config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            span = self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC)
        elif self._config.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            span = self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC)
        else:
            raise OneLoggerError(f"Invalid checkpoint strategy: {self._config.save_checkpoint_strategy}")
        self._training_state.multi_iteration_timers[span.name].stop(stop_time=stop_time)  # type: ignore[reportArgumentType]

        # Step 2: send an event of type SYNC_CHECKPOINT_METRICS_UPDATE if sync checkpoint.
        if self._config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            self.event(span, self.create_sync_checkpoint_metrics_event())

        # Step 3: stop the span.
        self.stop(span=span, stop_time=stop_time)
