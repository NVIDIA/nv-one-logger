# SPDX-License-Identifier: Apache-2.0
"""Test utilities for the wandb exporter module."""

from nv_one_logger.core.time import TracingTimestamp


def advance_time(time: TracingTimestamp, seconds: float) -> TracingTimestamp:
    """Advance a timestamp by the specified number of seconds."""
    return TracingTimestamp.for_timestamp(
        timestamp_sec=time.seconds_since_epoch + seconds,
        perf_counter=time.perf_counter_seconds + seconds,
        validate_timestamp=False,
    )
