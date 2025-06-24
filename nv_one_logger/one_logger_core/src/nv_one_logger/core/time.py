# SPDX-License-Identifier: Apache-2.0
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from nv_one_logger.core.exceptions import assert_that


@dataclass(frozen=True)
class TracingTimestamp:
    """
    A class uses to keep track of timestamp of an event.

    This class is useful for performance tracing, as it
    - is timezone agnostic.
    - is high resolution as it uses a performance counter for accurate calculation of the time elapsed.
    - has a regular timestamp for the wall clock time (useful for human readability).

    See https://superfastpython.com/time-time-vs-time-perf_counter/ for more details.
    """

    # The time in seconds since the epoch as a floating-point number. This is wall clock time and is same as time.time()
    _seconds_since_epoch: float

    # The value (in fractional seconds) of a performance counter, i.e. a clock with the highest available resolution to measure a short duration.
    # The reference point of the returned value is undefined, so that only the difference between the results of two calls is valid.
    _perf_counter_seconds: float

    @property
    def seconds_since_epoch(self) -> float:
        """
        Get the time in seconds since the epoch.

        Returns:
            float: The time since the epoch in fractional seconds.
        """
        return self._seconds_since_epoch

    @property
    def milliseconds_since_epoch(self) -> int:
        """
        Get the time in milliseconds since the epoch.

        Returns:
            float: The time since the epoch in milliseconds.
        """
        return int(self.seconds_since_epoch * 1000)

    @property
    def perf_counter_seconds(self) -> float:
        """
        Get the value of the performance counter in fractional seconds.

        Returns:
            float: The value of the performance counter in fractional seconds
        """
        return self._perf_counter_seconds

    @classmethod
    def now(cls) -> "TracingTimestamp":
        """
        Create a new TracingTimestamp instance representing the current time.

        Returns:
            TracingTimestamp: A new instance of TracingTimestamp with the current time
        """
        return cls(_seconds_since_epoch=time.time(), _perf_counter_seconds=time.perf_counter())

    @classmethod
    def for_timestamp(cls, timestamp_sec: float, perf_counter: Optional[float] = None, validate_timestamp: bool = True) -> "TracingTimestamp":
        """
        Create a new TracingTimestamp based on a given timestamp.

        Note that this function is menat for creating tracing timestamps, so in almost all cases, the timestamp should be very
        close to the current time. We use a threshold of 10 hours to allow for delayed reporting of spans/events for long running span.
        Enforcing this constraint helps us catch bugs in the code that use this function (e.g., using microseconds instead of fractional
        seconds or passing a perf counter value instead of a timestamp). If you have a legitimate case for creating a tracing timestamp
        that is not close to the current time, you can set validate_timestamp to false.

        Args:
            timestamp_sec (float): The timestamp in seconds since the epoch to create the TracingTimestamp from.

            perf_counter_seconds (float):Optional. The value of the performance counter in fractional seconds to create the TracingTimestamp from.
            If not provided, the perf counter value will be calculated based on the difference between the current wall clock time and the given timestamp.
            This may result in a slight loss of precision as it uses only the wall clock time to create the TracingTimestamp. If you need high precision,
            you need to know the value of time.time() as well as time.perf_counter() at the time of the event.

            validate_timestamp (bool): Whether to validate the timestamp. If set to true, the timestamp will be validated against the current time
            and rejected if it is not within 10 hours of the curent time. See the function's docstring for more details.
            Returns:
                TracingTimestamp: A new instance of TracingTimestamp created from the given timestamp.
        """
        if not perf_counter:
            # Let's guess the value of the perf counter at the time of the event. This should be fairly accurate
            # if the system clock has not been changed since the event. In general, the precision of the time.perf_counter() function
            # may be relatively higher than the precision of  clocks used by other time functions in the time module,
            # such as the time.time() function.
            perf_counter = time.perf_counter() - (time.time() - timestamp_sec)

        if validate_timestamp:
            threashold = 60 * 10  # 10 hours. This allows delayed reporting of spans/events for long running span.
            assert_that(timestamp_sec <= time.time() + threashold, "timestamp is in the future. This is meant to be the time in seconds since epoch.")
            assert_that(timestamp_sec >= time.time() - threashold, "timestamp is too old. This is meant to be the time in seconds since epoch.")
        return cls(_seconds_since_epoch=timestamp_sec, _perf_counter_seconds=perf_counter)

    def to_json(self) -> Dict[str, float]:
        """Convert the TracingTimestamp to a JSON-compatible dictionary that can be passed to json.dumps.

        Returns:
            dict: a JSON-compatible dictionary representation of the TracingTimestamp.
        """
        return {"seconds_since_epoch": self.seconds_since_epoch, "perf_counter_seconds": self.perf_counter_seconds}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TracingTimestamp":
        """Create a TracingTimestamp instance from a JSON-compatible dictionary (e.g., the return value of json.loads).

        Args:
            data: Dictionary containing the TracingTimestamp data.

        Returns:
            Event: A TracingTimestamp created from the data.
        """
        assert_that(data, "TracingTimestamp data cannot be None.")
        assert_that(
            data.get("seconds_since_epoch") and data.get("perf_counter_seconds"),
            f"TracingTimestamp must have seconds_since_epoch and perf_counter_seconds: {data}",
        )

        return cls(_seconds_since_epoch=data["seconds_since_epoch"], _perf_counter_seconds=data["perf_counter_seconds"])

    def __hash__(self) -> int:
        """
        Compute the hash value of this TracingTimestamp.

        Returns:
            int: A hash value based on both the epoch seconds and performance counter seconds.
        """
        return hash((self._seconds_since_epoch, self._perf_counter_seconds))

    def __eq__(self, other: object) -> bool:
        """
        Compare this TracingTimestamp with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is a TracingTimestamp with the same values,
                  False otherwise.
        """
        if not isinstance(other, TracingTimestamp):
            return False
        return self._seconds_since_epoch == other._seconds_since_epoch and self._perf_counter_seconds == other._perf_counter_seconds


class Timer:
    """
    A class representing the duration of a span/event.

    It can be used to measure the duration with high precision.
    """

    _start_time: Optional[TracingTimestamp] = None
    _elapsed_time_since_last_reset: float = 0.0
    _running: bool = False

    @property
    def running(self) -> bool:
        """Get the running state of the timer."""
        return self._running

    def start(self, start_time: Optional[TracingTimestamp] = None) -> None:
        """
        Start the duration measurement.

        Args:
            start_time: The time to start the duration measurement. If not provided, the current time will be used.
        """
        if not start_time:
            start_time = TracingTimestamp.now()
        assert_that(not self._running, "start called when the timer was already running")
        self._start_time = start_time
        self._running = True

    def stop(self, reset: bool = False) -> float:
        """
        Stop the duration measurement.

        Args:
            reset: Whether to reset the duration measurement. If set to true, a subsequent call to start()
            will reset the elapsed time to 0; otherwise, calling start() again will resume the timer and
            the elapsed time will be accumulated.

        Returns:
            The elapsed time since the start of the timer.
        """
        assert_that(self._running and self._start_time, "stop called when the timer was not running")
        self._running = False
        self._elapsed_time_since_last_reset += time.perf_counter() - self._start_time.perf_counter_seconds  # type: ignore[union-attr]
        if reset:
            ret = self._elapsed_time_since_last_reset
            self.reset()
            return ret
        else:
            return self._elapsed_time_since_last_reset

    def reset(self, start: bool = False) -> None:
        """
        Reset the duration and optionally re-start it.

        Args:
            start_time: The time to start the duration measurement, if start is True. If not provided, the current time will be used.
            start: Whether to start the duration measurement immediately.
        """
        self._start_time = None
        self._elapsed_time_since_last_reset = 0.0
        self._running = False
        if start:
            self.start()
