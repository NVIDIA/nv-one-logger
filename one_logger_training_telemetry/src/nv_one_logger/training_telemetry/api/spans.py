# SPDX-License-Identifier: Apache-2.0
from strenum import StrEnum


class StandardTrainingJobSpanName(StrEnum):
    """
    Enum class representing names of predefined spans.

    See README.md for more details on the organization of the spans.
    """

    # ***********************************************************
    # Long running spans
    # ***********************************************************#

    # The training loop.
    TRAINING_LOOP = "training_loop"

    # The validation loop.
    VALIDATION_LOOP = "validation_loop"

    # The testing loop.
    TESTING_LOOP = "testing_loop"

    # ***********************************************************#
    # Initialization spans
    # ***********************************************************#

    # The initialization of the distributed training code.
    DIST_INIT = "distributed_code_initialization"
    # The initialization of the data loader.
    DATA_LOADER_INIT = "data_loader_initialization"
    # The initialization of the model.
    MODEL_INIT = "model_initialization"
    # The initialization of the optimizer.
    OPTIMIZER_INIT = "optimizer_initialization"

    # ***********************************************************#
    # Checkpoint spans
    # ***********************************************************#

    # The loading of a checkpoint.
    CHECKPOINT_LOAD = "checkpoint_load"
    # The saving of a checkpoint synchronously. This includes the entire duration of saving the checkpoint.
    CHECKPOINT_SAVE_SYNC = "checkpoint_save_sync"
    # The saving of a checkpoint asynchronously.
    # This includes only the time that it took to trigger the async saving (the time that the training loop was blocked for).
    # The actual saving of the checkpoint is done in a separate thread/process and is not included in this span.
    CHECKPOINT_SAVE_ASYNC = "checkpoint_save_async"
    # The finalization of an asynchronous checkpoint save.
    # This includes the work done by the async saving thread/process.
    CHECKPOINT_SAVE_FINALIZATION = "checkpoint_save_finalization"

    # ***********************************************************#
    # Very short spans, normally used for profiling backends
    # ***********************************************************#

    # A single training iteration, also known as a training step.
    TRAINING_SINGLE_ITERATION = "training_single_iteration"
    # A single validation iteration, also known as a validation step.
    VALIDATION_SINGLE_ITERATION = "validation_single_iteration"
    # A single test iteration, also known as a test step.
    TESTING_SINGLE_ITERATION = "TESTING_SINGLE_ITERATION"
    # The loading of data inside a training iteration.
    DATA_LOADING = "data_loading"
    # The forward pass inside a training iteration.
    MODEL_FORWARD = "model_forward"
    # The zeroing of the gradients.
    ZERO_GRAD = "zero_grad"
    # The backward pass inside a training iteration.
    MODEL_BACKWARD = "model_backward"
    # The update of the optimizer.
    OPTIMIZER_UPDATE = "optimizer_update"
