# SPDX-License-Identifier: Apache-2.0
from strenum import StrEnum


class StandardTrainingJobEventName(StrEnum):
    """
    Enum class representing different names of predefined training events.

    See README.md for more details on the organization of the spans and their events.
    """

    # ***********************************************************
    # Predefined events for the APPLICATION span
    # ***********************************************************
    # An event representing the initialization of the OneLogger library. This event is reported as part of the APPLICATION span.
    ONE_LOGGER_INITIALIZATION = "one_logger_initialization"

    # An event representing an update to the training telemetry configuration. This event is reported as part of the APPLICATION span
    # whenever the training telemetry config is updated or becomes available.
    UPDATE_TRAINING_TELEMETRY_CONFIG = "update_training_telemetry_config"

    # ***********************************************************
    # Predefined events for the TRAINING_LOOP span
    # ###########################################################
    # An event representing multiple training iterations reported in aggregated form. This is normally used for logging
    # application progress in a training loop. That is this event is fired as part of the TRAINING_LOOP span periodically
    # (after completion of N iterations, where N equals the "log_every_n_train_iterations" config knob). The event contains
    # average values for metrics across N iterations. The event timestamp reflects when the iterations are logged, not when they were executed.
    # The reason we need this event is because the training loop is too long as a single span and a single training iteration
    # may be too fine grained (resulting in exporting too much data). So we report some of the training metrics every N
    # iterations and include those in an event of type TRAINING_MULTI_ITERATION.
    # For validation and testing, we can report the metrics at the end of each validation or test span, as these spans are
    # often short.
    TRAINING_METRICS_UPDATE = "training_metrics_update"

    # ***********************************************************
    # Predefined events for the VALIDATION_LOOP span
    # ***********************************************************
    # An event representing the validation metrics update. This event is reported as part of the VALIDATION_LOOP span.
    # The validation span is often short, we normally have a single event of type VALIDATION_METRICS_UPDATE within that span.
    # We use an event instead of attaching the aggregate metrics to the span for consistency with how we do things in training
    # as well as allowing to send these updates more frequently (more than once per each VALIDATION_LOOP span) in future if needed
    VALIDATION_METRICS_UPDATE = "validation_metrics_update"

    # ***********************************************************
    # Predefined events for the TESTING_LOOP span
    # ***********************************************************
    # An event representing the test metrics update. This event is reported as part of the TESTING_LOOP span.
    # The test span is often short, we normally have a single event of type TESTING_METRICS_UPDATE within that span.
    # We use an event instead of attaching the aggregate metrics to the span for consistency with how we do things in training
    # as well as allowing to send these updates more frequently (more than once per each TESTING_LOOP span) in future if needed
    TESTING_METRICS_UPDATE = "testing_metrics_update"

    # ***********************************************************
    # Predefined events for the CHECKPOINT_SAVE_ASYNC or
    # CHECKPOINT_SAVE_SYNC span
    # ***********************************************************
    # An event representing the successful saving of a checkpoint. This event is reported as part of
    # CHECKPOINT_SAVE_SYNC span (for sync checkpoint saving) or the TRAINING_LOOP span (for async checkpoint saving).
    # This is because for async checkpoint saving, the StandardTrainingJobSpanName.CHECKPOINT_SAVE_ASYNC span represents the
    # action of scheduling the checkpoint save and the span may end before the checkpoint is actually saved. That's why
    # the SAVE_CHECKPOINT_SUCCESS event is added to the TRAINING_LOOP span for async checkpoint saving.
    SAVE_CHECKPOINT_SUCCESS = "save_checkpoint_success"

    # An event representing an update to the timing information for the sync checkpoint save operations.
    # This event is reported as part of the CHECKPOINT_SAVE_SYNC span.
    # Note: This event is not reported for async checkpoint saving because the timing information for
    # async checkpoint saving consists only of the time taken to start the save operation. The creation and
    # saving of an async checkpoint is done in a separate thread/process and doesn't affect the training time.
    SYNC_CHECKPOINT_METRICS_UPDATE = "sync_checkpoint_metrics_update"
