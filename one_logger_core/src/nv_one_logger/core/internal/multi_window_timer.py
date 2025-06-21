# SPDX-License-Identifier: Apache-2.0
"""Contains the MultiWindowTimer class."""

from typing import Optional

from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.time import TracingTimestamp


class MultiWindowTimer:
    """
    A class for tracking timing windows and their statistics.

    This class allows for the tracking the duration of an operation that occurs over
    multiple timing windows. That is, the operation may stop and start (resume) multiple
    times with periods of inactivity (pause) between them. This timer keeps track
    of each time window in which the operation is active, and provides properties to
    access the total time, minimum, maximum, and average duration of all completed
    windows.

    Here is an example: Imagine that you have a long-running job that loads a batch of
    data from a data lake, transforms it, saves transformed data, then loads the next
    batch of data and repeats the same process (and this continues until all data is
    processed).

    If we are interested in knowing the total time spent on loading data as well as
    min/max/avg of loading each batch of data, we can use this class, call start()
    every time we start loading a batch of data, and call stop() every tine we finish
    loading a batch of data. In this example, a "window" is the duration of time that
    the job spends on loading a batch of data (the period between each subsequent
    call to start() and stop()).

    Note that this class is not thread-safe and also it assumes that your activity
    windows are not overlapping (you finish one window before starting another).
    """

    def __init__(self) -> None:
        """Initialize a new MultiWindowTimer."""
        # Time info of a few important windows.
        self._current_window_start: Optional[TracingTimestamp] = None
        self._first_window_start: Optional[TracingTimestamp] = None
        self._latest_window_end: Optional[TracingTimestamp] = None
        self._latest_window_duration_sec: float = 0.0

        # Number of windows so far (including both active and completed windows).
        self._total_window_count: int = 0

        # Total time across all windows (does not include the time between a stop call and the next start call as that time is not part of any window).
        self._total_time_sec: float = 0.0

        # Min/max/avg duration of a window.
        self._min_window_duration_sec: float = float("inf")
        self._max_window_duration_sec: float = 0.0
        self._avg_window_duration_sec: float = 0.0

    def start(self, start_time: Optional[TracingTimestamp] = None) -> None:
        """
        Start a new timing window.

        Args:
            start_time: The start time of the window. If None, the current time is used.

        Raises:
            OneLoggerError: If the timer is already active.
        """
        if not start_time:
            start_time = TracingTimestamp.now()

        if self._current_window_start:
            raise OneLoggerError("Cannot start timer since it is already active")

        self._current_window_start = start_time
        self._total_window_count += 1

        if self._first_window_start is None:
            self._first_window_start = self._current_window_start

    def stop(self, stop_time: Optional[TracingTimestamp] = None) -> None:
        """
        Stop the current timing window and update statistics.

        Args:
            stop_time: The end time of the window. If None, the current time is used.

        Raises:
            OneLoggerError: If the timer is not active.
        """
        if not stop_time:
            stop_time = TracingTimestamp.now()

        try:
            if not self._current_window_start:
                raise OneLoggerError("Cannot stop timer since it is not active")
            if stop_time.seconds_since_epoch < self._current_window_start.seconds_since_epoch:
                raise OneLoggerError(
                    "Cannot stop timer with a stop time that is before the start time."
                    + f" stop_time: {stop_time}, current_window_start: {self._current_window_start}"
                )

            self._latest_window_end = stop_time
            self._latest_window_duration_sec = stop_time.perf_counter_seconds - self._current_window_start.perf_counter_seconds

            self._min_window_duration_sec = (
                min(self._min_window_duration_sec, self._latest_window_duration_sec)
                if self._min_window_duration_sec != float("inf")
                else self._latest_window_duration_sec
            )
            self._max_window_duration_sec = max(self._max_window_duration_sec, self._latest_window_duration_sec)

            self._total_time_sec += self._latest_window_duration_sec
            self._avg_window_duration_sec = self._total_time_sec / self._total_window_count
        finally:
            self._current_window_start = None

    def reset(self) -> None:
        """Reset the timer."""
        self._current_window_start = None

        self._first_window_start = None
        self._latest_window_end = None
        self._latest_window_duration_sec = 0

        self._total_window_count = 0
        self._total_time_sec = 0.0
        self._min_window_duration_sec = float("inf")
        self._max_window_duration_sec = 0.0
        self._avg_window_duration_sec = 0.0

    @property
    def is_active(self) -> bool:
        """Get whether the timer is currently active.

        An active timer is one that has been started but not yet stopped.

        Returns:
            bool: True if the timer is active, False otherwise.
        """
        return self._current_window_start is not None

    @property
    def current_window_start(self) -> Optional[TracingTimestamp]:
        """Get the start time of the current timing window.

        Returns:
            Optional[TracingTimestamp]: The start time of the current window, or None if no window is active.
        """
        return self._current_window_start

    @property
    def total_window_count(self) -> int:
        """Get the total number of windows so far (including both active and completed windows)."""
        return self._total_window_count

    @property
    def total_time_sec(self) -> float:
        """Get the total time across all timing windows.

        Returns:
            float: The total time in seconds.
        """
        return self._total_time_sec

    @property
    def min_window_duration_sec(self) -> float:
        """Get the minimum duration of any timing window.

        Returns:
            float: The minimum window duration in seconds.
        """
        return self._min_window_duration_sec

    @property
    def max_window_duration_sec(self) -> float:
        """Get the maximum duration of any timing window.

        Returns:
            float: The maximum window duration in seconds.
        """
        return self._max_window_duration_sec

    @property
    def avg_window_duration_sec(self) -> float:
        """Get the average duration of all timing windows.

        Returns:
            float: The average window duration in seconds.
        """
        return self._avg_window_duration_sec

    @property
    def latest_window_duration_sec(self) -> float:
        """Get the duration of the most recently completed timing window.

        Returns:
            float: The duration of the latest window in seconds.
        """
        return self._latest_window_duration_sec

    @property
    def first_window_start(self) -> Optional[TracingTimestamp]:
        """Get the start time of the first timing window.

        Returns:
            Optional[TracingTimestamp]: The start time of the first window, or None if no window has been started.
        """
        return self._first_window_start

    @property
    def latest_window_end(self) -> Optional[TracingTimestamp]:
        """Get the end time of the latest timing window.

        Returns:
            Optional[TracingTimestamp]: The end time of the latest window, or None if no window has been started.
        """
        return self._latest_window_end

    def __eq__(self, other: object) -> bool:
        """Compare this timer with another timer for equality."""
        if not isinstance(other, MultiWindowTimer):
            return False
        return (
            self._current_window_start == other._current_window_start
            and self._first_window_start == other._first_window_start
            and self._latest_window_end == other._latest_window_end
            and self._latest_window_duration_sec == other._latest_window_duration_sec
            and self._total_window_count == other._total_window_count
            and self._total_time_sec == other._total_time_sec
            and self._min_window_duration_sec == other._min_window_duration_sec
            and self._max_window_duration_sec == other._max_window_duration_sec
            and self._avg_window_duration_sec == other._avg_window_duration_sec
        )

    def __hash__(self) -> int:
        """Compute a hash value for this timer."""
        return hash(
            (
                self._current_window_start,
                self._first_window_start,
                self._latest_window_end,
                self._latest_window_duration_sec,
                self._total_window_count,
                self._total_time_sec,
                self._min_window_duration_sec,
                self._max_window_duration_sec,
                self._avg_window_duration_sec,
            )
        )
