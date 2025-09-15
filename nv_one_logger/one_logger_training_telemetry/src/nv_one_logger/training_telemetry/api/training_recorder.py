# SPDX-License-Identifier: Apache-2.0
"""
This file contains the TrainingRecorder class, which is responsible for recording training telemetry data.

The TrainingRecorder class extends the DefaultRecorder class and provides specialized recording capabilities
for training-related telemetry data.

"""
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, cast

from overrides import override  # type: ignore[ancereportUnknownVariableType]

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event
from nv_one_logger.core.exceptions import OneLoggerError, assert_that
from nv_one_logger.core.internal.metric_summarizer import MetricSummarizer
from nv_one_logger.core.internal.multi_window_timer import MultiWindowTimer
from nv_one_logger.core.internal.safe_execution import safely_execute
from nv_one_logger.core.internal.version import get_version
from nv_one_logger.core.span import Span, SpanName, StandardSpanName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.recorder.default_recorder import DefaultRecorder, ExportCustomizationMode
from nv_one_logger.training_telemetry.api.attributes import (
    CheckpointSaveSpanAttributes,
    OneLoggerInitializationAttributes,
    SaveCheckpointSuccessEventAttributes,
    SyncCheckpointMetricsUpdateAttributes,
    TestingMetricsUpdateAttributes,
    TrainingLoopAttributes,
    TrainingMetricsUpdateAttributes,
    TrainingTelemetryAttributes,
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
class _ProductivityState:
    """Stores productivity metrics as of a certain training iteration (global step)."""

    # Completed training iterations (including iterations from the loaded checkpoint, i.e.,
    # iterations before "train_iterations_start") where the work was saved to a checkpoint.
    productive_train_iterations: int

    # Number of training samples processed(including iterations from the loaded checkpoint, i.e.,
    # iterations before "train_iterations_start") where the work was saved to a checkpoint.
    productive_train_samples: int

    # Total time spent on training iterations that have been saved to a checkpoint (does NOT include the time
    # spent in another job whose checkpoint is loaded at start of this job).
    productive_train_iterations_sec: float

    # Total time spent on validation iterations for work that has been saved to a checkpoint (does NOT include the time
    # spent in another job whose checkpoint is loaded at start of this job).
    productive_validation_iterations_sec: float

    # Number of floating point operations completed so far (including the ones from the loaded checkpoint and
    # the ones from the current job) for work that has been saved to a checkpoint. None if unknown or unmeasured.
    productive_train_tflops: Optional[float] = None


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
    train_samples_processed_current_job: int = 0

    # Total number of floating point operations in the current job.
    total_flops_current_job: int = 0

    # Number of training tokens processed so far in the current job or None if the sequence length is not known.
    # None if unknown or unmeasured.
    train_tokens_current_job: Optional[int] = None

    # Number of floating point operations completed so far (including the ones from the loaded checkpoint and
    # the ones from the current job). None if unknown or unmeasured.
    completed_floating_point_operations_overall: Optional[int] = None

    # The timestamp of the start of the training loop.
    training_loop_start_time: Optional[TracingTimestamp] = None

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

    # The productivity state at a particular iteration.
    # When a training job fails, the work done between the last successful checkpoint and the time of failure is
    # wasted (non-productive) work. We keep track of the work done (number of completed iterations, samples processed, etc
    # right before we create a checkpoint and if the checkpoint succeeds, we consider that work to be productive.
    # Since we support async checkpoints, we cannot assume that the current state (current iteration, samples processed, etc)
    # at the time of the checkpoint save success reflects what was included in the checkpoint (we may do additional work
    # after kicking off an async checkpoint). This is why this field is a dictionary that maps iteration number (global step)
    # at the time the checkpoint save operation started to a _ProductivityState object.
    productivity_state: Dict[int, _ProductivityState] = field(default_factory=dict)

    # The maximum iteration number (global step) for which productivty metrics were reported.
    # This is used to avoid reporting productivity metrics for an older checkpoint (which could happen
    # with async checkpoints if they complete out of order).
    max_reported_productive_train_iterations: int = -1


class TrainingRecorder(DefaultRecorder):
    """A recorder specifically designed for training telemetry.

    This class extends DefaultRecorder to provide specialized recording capabilities
    for training-related telemetry data.
    """

    def __init__(
        self,
        config: OneLoggerConfig,
        exporters: List[Exporter],
        export_customization_mode: ExportCustomizationMode,
        span_name_filter: Optional[List[SpanName]],
    ):
        """Initialize the TrainingRecorder with a list of exporters.

        Args:
            config: The base OneLogger configuration (which may contain training telemetry config).
            exporters: A list of exporters to use for recording training telemetry.
            export_customization_mode: The mode of exporting spans (and their associated events and attribytes) to exporters.
            span_name_filter: This argument should be interpretted wrt the value of export_customization_mode:
                If export_customization_mode is ExportCustomizationMode.EXPORT_ALL_SPANS, span_name_filter should not be set.
                If export_customization_mode is ExportCustomizationMode.WHITELIST_SPANS, span_name_filter is a list of span names to export (whitelist).
                If export_customization_mode is ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter is a list of span names to not export (blacklist).
        """
        super().__init__(exporters, export_customization_mode=export_customization_mode, span_name_filter=span_name_filter)

        self._config: OneLoggerConfig = config

        self._training_state = _TrainingState()

    def _get_training_config(self) -> TrainingTelemetryConfig:
        """Get the training telemetry config, ensuring it's the correct type.

        Returns:
            TrainingTelemetryConfig: The training telemetry configuration.

        Raises:
            OneLoggerError: If telemetry_config is None or not a TrainingTelemetryConfig.
        """
        if self._config.telemetry_config is None:
            raise OneLoggerError(
                "Training telemetry config must be set before the start of training. "
                "See the api for TrainingTelemetryProvider.set_training_telemetry_config for more details."
            )

        return self._config.telemetry_config

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

        if span.name in self._training_state.multi_iteration_timers.keys():
            timer = self._training_state.multi_iteration_timers[span.name]
            # Fail-safe timer stops here to prevent unexpected exits if the timer is not stopped by the user.
            if timer.is_active:
                timer.stop(stop_time)

                # Record an error event on the span to indicate that the timer was forced to stop
                self.error(
                    span=span,
                    error_message=(
                        f"Timer for span {span.name} was automatically stopped because the span is being stopped. "
                        "This may indicate that the corresponding on_xxx_end method was not called correctly. "
                        "If the program exited properly, please double-check your on_xxx_end callback. "
                        "If the program exited abnormally (e.g., due to an exception), this is expected."
                    ),
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

        # Create attributes with base config fields only
        attributes = OneLoggerInitializationAttributes.create(
            one_logger_training_telemetry_version=get_version("nv-one-logger-training-telemetry"),
            enable_for_current_rank=conf.enable_for_current_rank,
            session_tag=conf.session_tag,
            world_size=conf.world_size,
            is_baseline_run=conf.is_baseline_run,
            summary_data_schema_version=conf.summary_data_schema_version,
            rank=_get_rank(),
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

        # If training config is available, update the application span with training telemetry configuration
        if self._config.telemetry_config:
            self._update_application_span_with_training_telemetry_config(training_telemetry_config=self._config.telemetry_config)

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

    def on_distributed_init_start(self, start_time: TracingTimestamp) -> Span:
        """Start a new span for distributed initialization.

        Args:
            start_time: The timestamp of the start of distributed initialization.

        Returns:
            Span: The newly created span for distributed initialization.
        """
        return self.start(
            span_name=StandardTrainingJobSpanName.DIST_INIT,
            start_time=start_time,
        )

    def on_distributed_init_end(self, stop_time: TracingTimestamp) -> None:
        """Stop the distributed initialization span, and update the state if necessary.

        Args:
            stop_time: The timestamp of the end of distributed initialization.
        """
        self.stop(
            span=self._get_active_span(StandardTrainingJobSpanName.DIST_INIT),
            stop_time=stop_time,
        )

    @safely_execute
    def _update_application_span_with_training_telemetry_config(self, training_telemetry_config: TrainingTelemetryConfig) -> None:
        """Update the application span with training telemetry configuration when training config becomes available.

        Args:
            training_telemetry_config: The training telemetry config.
        """
        if training_telemetry_config is None:
            raise OneLoggerError("Please set the training telemetry config before calling this method.")

        timestamp = TracingTimestamp.now()

        # Check if application span is active
        app_spans = self.get_active_spans_by_name(StandardSpanName.APPLICATION)
        if len(app_spans) == 0:
            raise OneLoggerError("Cannot update training metrics: Please call on_app_start() before calling this method.")

        app_span = app_spans[0]

        if training_telemetry_config.is_log_throughput_enabled:
            self._training_state.completed_floating_point_operations_overall = 0

        # Create training telemetry configuration attributes
        training_params_attributes = TrainingTelemetryAttributes.create(
            perf_tag=training_telemetry_config.perf_tag,
            global_batch_size=training_telemetry_config.global_batch_size,
            log_every_n_train_iterations=training_telemetry_config.log_every_n_train_iterations,
            micro_batch_size=training_telemetry_config.micro_batch_size,
            seq_length=training_telemetry_config.seq_length,
            flops_per_sample=training_telemetry_config.flops_per_sample,
            train_iterations_target=training_telemetry_config.train_iterations_target,
            train_samples_target=training_telemetry_config.train_samples_target,
            checkpoint_strategy=training_telemetry_config.save_checkpoint_strategy,
            is_train_iterations_enabled=training_telemetry_config.is_train_iterations_enabled,
            is_validation_iterations_enabled=training_telemetry_config.is_validation_iterations_enabled,
            is_test_iterations_enabled=training_telemetry_config.is_test_iterations_enabled,
            is_save_checkpoint_enabled=training_telemetry_config.is_save_checkpoint_enabled,
            is_log_throughput_enabled=training_telemetry_config.is_log_throughput_enabled,
            custom_metadata=training_telemetry_config.custom_metadata,
        )

        # Create and post the update training telemetry config event
        self.event(
            app_span,
            Event.create(
                name=StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG,
                attributes=training_params_attributes,
                timestamp=timestamp,
            ),
        )

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
        training_telemetry_config = self._get_training_config()

        if not start_time:
            start_time = TracingTimestamp.now()

        # Step 1: Update the state.
        state = self._training_state
        state.training_loop_start_time = start_time
        state.train_iterations_start = train_iterations_start
        state.train_samples_start = train_samples_start
        # We assume the first iteration is iteration 0. So completed_training_iterations_overall is the same as train_iterations_start.
        state.completed_training_iterations_overall = train_iterations_start
        # iteration number (global step) is zero-based. So if completed_training_iterations_overall is N, the next training iteration will be iteration N.
        state.validation_interval_start = state.completed_training_iterations_overall
        state.testing_interval_start = state.completed_training_iterations_overall
        if training_telemetry_config.is_log_throughput_enabled:
            assert_that(
                training_telemetry_config.flops_per_sample and training_telemetry_config.flops_per_sample > 0,
                "flops_per_sample must be set to a positive value when is_log_throughput_enabled is True",
            )
            # The initial value of completed_floating_point_operations_overall is nonzero if loading ckpt, whereas total_flops_current_job
            # is always initialized to zero. For example, if train_iterations_start is 1,  it means that one iteration (iteration 0) has
            # been completed in a previous run.
            state.completed_floating_point_operations_overall = cast(
                int, train_iterations_start * training_telemetry_config.global_batch_size * training_telemetry_config.flops_per_sample  # type: ignore
            )
            state.total_flops_current_job = 0

        # Step 2: Create the span.
        # Training telemetry configuration (perf_tag, global_batch_size, etc.) are now posted
        # via UPDATE_TRAINING_TELEMETRY_CONFIG event in the application span, so we only include
        # training-loop-specific attributes here
        span_attributes = TrainingLoopAttributes.create(
            train_iterations_start=train_iterations_start,
            train_samples_start=train_samples_start,
            train_tokens_target=train_tokens_target,
            completed_floating_point_operations_overall=state.completed_floating_point_operations_overall,
            train_iterations_target=train_iterations_target,
            train_samples_target=train_samples_target,
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
        training_telemetry_config = self._get_training_config()

        # Step 1: Update the state.
        training_iteration_timer = self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        training_iteration_timer.stop(stop_time)

        self._training_state.completed_training_iterations_overall += 1
        self._training_state.train_samples_processed_current_job += training_telemetry_config.global_batch_size
        if training_telemetry_config.seq_length:
            self._training_state.train_tokens_current_job = training_telemetry_config.seq_length * self._training_state.train_samples_processed_current_job

        self._training_state.last_logged_train_iterations_finish_time = stop_time
        if not self._training_state.first_logged_train_iterations_finish_time:
            self._training_state.first_logged_train_iterations_finish_time = stop_time

        if training_telemetry_config.is_log_throughput_enabled:
            assert_that(
                training_telemetry_config.flops_per_sample and training_telemetry_config.flops_per_sample > 0,
                "flops_per_sample must be set to a positive value when is_log_throughput_enabled is True",
            )
            flops = training_telemetry_config.global_batch_size * training_telemetry_config.flops_per_sample  # type: ignore[reportOperatorIssue]
            assert_that(
                self._training_state.completed_floating_point_operations_overall is not None, "completed_floating_point_operations_overall must be initialized."
            )
            self._training_state.completed_floating_point_operations_overall += flops  # type: ignore[reportOperatorIssue]
            self._training_state.total_flops_current_job += flops
            train_iterations_time_total = training_iteration_timer.total_time_sec
            assert_that(train_iterations_time_total > 0, "train_iterations_time_total must be greater than 0")
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
        # We are adding 1 to latest_iteration to make this compatible with the previous implementation, which logged
        # metrics on the iteration before iterations that are multiple of log_every_n_train_iterations.
        training_telemetry_config = self._get_training_config()
        log_every_n_train_iterations = training_telemetry_config.log_every_n_train_iterations
        if latest_iteration > 0 and (latest_iteration + 1) % log_every_n_train_iterations == 0:
            training_loop_span = self._get_active_span(StandardTrainingJobSpanName.TRAINING_LOOP)
            training_iteration_timer = self._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
            attributes = TrainingMetricsUpdateAttributes.create(
                train_iterations_start=self._training_state.train_iterations_start,
                current_iteration=latest_iteration,
                num_iterations=training_iteration_timer.total_window_count,
                train_samples_start=self._training_state.train_samples_start,
                num_train_samples=self._training_state.train_samples_processed_current_job,
                interval=log_every_n_train_iterations,
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
                last_logged_train_iterations_finish_timestamp_sec=(
                    self._training_state.last_logged_train_iterations_finish_time.seconds_since_epoch
                    if self._training_state.last_logged_train_iterations_finish_time
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

    def create_sync_checkpoint_metrics_event(self, span_name: StandardTrainingJobSpanName) -> Event:
        """Create an event of type SYNC_CHECKPOINT_METRICS_UPDATE using the most recent checkpoint metrics.

        Although the event name says "SYNC", this captures the main-thread window (start→end) for checkpoint
        saving regardless of strategy (SYNC or ASYNC). The appropriate timer is selected based on the span.
        """
        checkpoint_timer = self._training_state.multi_iteration_timers[span_name]
        attributes = SyncCheckpointMetricsUpdateAttributes.create(
            save_checkpoint_sync_time_total_sec=checkpoint_timer.total_time_sec,
            save_checkpoint_sync_time_min_sec=checkpoint_timer.min_window_duration_sec,
            save_checkpoint_sync_time_max_sec=checkpoint_timer.max_window_duration_sec,
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
        training_telemetry_config = self._get_training_config()

        span_name = None
        if training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            span_name = StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC
        elif training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            span_name = StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC
        else:
            raise OneLoggerError(f"Invalid checkpoint strategy: {training_telemetry_config.save_checkpoint_strategy}")
        if training_telemetry_config.is_save_checkpoint_enabled:
            timers = self._training_state.multi_iteration_timers
            self._training_state.productivity_state[current_iteration] = _ProductivityState(
                productive_train_iterations=self._training_state.completed_training_iterations_overall,
                productive_train_samples=self._training_state.train_samples_start + self._training_state.train_samples_processed_current_job,
                productive_train_iterations_sec=timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION].total_time_sec,
                productive_validation_iterations_sec=timers[StandardTrainingJobSpanName.VALIDATION_SINGLE_ITERATION].total_time_sec,
                productive_train_tflops=(
                    float(self._training_state.completed_floating_point_operations_overall) / (10**12)
                    if self._training_state.completed_floating_point_operations_overall is not None
                    else None
                ),
            )

            timer = self._training_state.multi_iteration_timers[span_name]
            timer.start(start_time)

            # Step 2: Create the span.
            span_attributes = CheckpointSaveSpanAttributes.create(
                training_telemetry_config.save_checkpoint_strategy,
                current_iteration=current_iteration,
                # The current save attempt is already included in the total window count.
                save_checkpoint_attempt_count=timer.total_window_count,
            )
        else:
            span_attributes = None
        return self.start(span_name=span_name, span_attributes=span_attributes)

    def on_save_checkpoint_success(self, current_iteration: int, timestamp: TracingTimestamp) -> None:
        """Send an event of type SAVE_CHECKPOINT_SUCCESS and update the state if necessary.

        Args:
            current_iteration: The current iteration number (global step) at the time the checkpoint save operation started.
            timestamp: The timestamp of the checkpoint saving.
        """
        training_telemetry_config = self._get_training_config()
        if not training_telemetry_config.is_save_checkpoint_enabled:
            return

        # Step 1: Update the state.
        parent_span = None
        # See comments on StandardTrainingJobEventName.SAVE_CHECKPOINT_SUCCESS.
        if training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            parent_span = self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC)
        elif training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            parent_span = self._get_active_span(StandardSpanName.APPLICATION)
        else:
            raise OneLoggerError(f"Invalid checkpoint strategy: {training_telemetry_config.save_checkpoint_strategy}")

        state = self._training_state
        state.successful_save_checkpoint_count_current_job += 1
        state.latest_save_checkpoint_success_time = timestamp
        if not state.first_save_checkpoint_success_time:
            state.first_save_checkpoint_success_time = timestamp

        productivity_state = state.productivity_state.get(current_iteration)
        if productivity_state is not None:
            self._training_state.productivity_state.pop(current_iteration)
        else:
            # This shouldn't happen but just in case...
            productivity_state = _ProductivityState(
                productive_train_iterations=0,
                productive_train_samples=0,
                productive_train_iterations_sec=0,
                productive_validation_iterations_sec=0,
                productive_train_tflops=0,
            )

        # Step 2: Create the event.
        event_attributes = SaveCheckpointSuccessEventAttributes.create(
            checkpoint_strategy=training_telemetry_config.save_checkpoint_strategy,
            current_iteration=current_iteration,
            first_successful_save_checkpoint_timestamp_sec=state.first_save_checkpoint_success_time.seconds_since_epoch,
            latest_successful_save_checkpoint_timestamp_sec=state.latest_save_checkpoint_success_time.seconds_since_epoch,
            save_checkpoint_success_count=state.successful_save_checkpoint_count_current_job,
            productive_train_iterations=productivity_state.productive_train_iterations,
            productive_train_samples=productivity_state.productive_train_samples,
            productive_train_iterations_sec=productivity_state.productive_train_iterations_sec,
            productive_validation_iterations_sec=productivity_state.productive_validation_iterations_sec,
            training_start_timestamp_sec=(state.training_loop_start_time.seconds_since_epoch if state.training_loop_start_time else None),
            productive_train_tflops=productivity_state.productive_train_tflops,
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
        span = None
        training_telemetry_config = self._get_training_config()
        if training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC:
            span = self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_SAVE_SYNC)
        elif training_telemetry_config.save_checkpoint_strategy == CheckPointStrategy.ASYNC:
            span = self._get_active_span(StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC)
        else:
            raise OneLoggerError(f"Invalid checkpoint strategy: {training_telemetry_config.save_checkpoint_strategy}")

        # Step 1: Update the state.
        if training_telemetry_config.is_save_checkpoint_enabled:
            self._training_state.multi_iteration_timers[span.name].stop(stop_time=stop_time)  # type: ignore[reportArgumentType]

            # Step 2: send an event of type SYNC_CHECKPOINT_METRICS_UPDATE for both sync and async strategies.
            self.event(span, self.create_sync_checkpoint_metrics_event(span.name))

        # Step 3: stop the span.
        self.stop(span=span, stop_time=stop_time)
