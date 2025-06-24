# SPDX-License-Identifier: Apache-2.0
"""Module for checkpoint-related enums and classes."""

from strenum import StrEnum


class CheckPointStrategy(StrEnum):
    """The strategy of checkpointing."""

    SYNC = "sync"
    ASYNC = "async"


class CheckPointType(StrEnum):
    """The type of checkpoint."""

    # Global checkpoint, saved to remote storage, and persistent.
    GLOBAL = "global"
    # Local checkpoint, saved to local storage, and non-persistent.
    LOCAL = "local"
