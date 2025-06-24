# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MultiWindowTimer class."""

from unittest.mock import Mock, patch

import pytest

from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.multi_window_timer import MultiWindowTimer
from nv_one_logger.core.time import TracingTimestamp


def test_init() -> None:
    """Test that a new MultiWindowTimer is initialized with correct default values."""
    timer = MultiWindowTimer()
    assert timer.current_window_start is None
    assert timer.total_window_count == 0
    assert timer.total_time_sec == 0.0
    assert timer.min_window_duration_sec == float("inf")
    assert timer.max_window_duration_sec == 0.0
    assert timer.avg_window_duration_sec == 0.0
    assert timer.latest_window_duration_sec == 0.0


@patch("time.time")
@patch("time.perf_counter")
def test_start(mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that starting the timer sets the correct state."""
    mock_time.return_value = 1000.0
    mock_perf_counter.return_value = 10.0
    timer = MultiWindowTimer()
    timer.start()
    assert timer.current_window_start is not None
    assert timer.current_window_start.seconds_since_epoch == 1000.0
    assert timer.current_window_start.perf_counter_seconds == 10.0
    assert timer.total_window_count == 1
    assert timer.first_window_start == TracingTimestamp.for_timestamp(timestamp_sec=1000.0, perf_counter=10.0, validate_timestamp=False)
    assert timer.latest_window_end is None


@patch("time.time")
@patch("time.perf_counter")
def test_with_explicit_timestamps(mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that using the timer with explicit timestamps."""
    mock_time.return_value = 1000.0
    mock_perf_counter.return_value = 10.0
    timer = MultiWindowTimer()
    timer.start(TracingTimestamp.for_timestamp(timestamp_sec=5555, perf_counter=40, validate_timestamp=False))
    assert timer.current_window_start is not None
    assert timer.current_window_start.seconds_since_epoch == 5555.0
    assert timer.current_window_start.perf_counter_seconds == 40.0
    assert timer.total_window_count == 1
    assert timer.first_window_start == TracingTimestamp.for_timestamp(timestamp_sec=5555.0, perf_counter=40.0, validate_timestamp=False)
    assert timer.latest_window_end is None

    timer.stop(TracingTimestamp.for_timestamp(timestamp_sec=5585, perf_counter=70, validate_timestamp=False))
    assert not timer.current_window_start
    assert timer.total_window_count == 1
    assert timer.total_time_sec == 30.0
    assert timer.min_window_duration_sec == 30.0
    assert timer.max_window_duration_sec == 30.0
    assert timer.avg_window_duration_sec == 30.0
    assert timer.latest_window_duration_sec == 30.0


@patch("time.time")
@patch("time.perf_counter")
def test_invalid_stop(mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that stopping the timer with a stop time that is before the start time raises an error."""
    timer = MultiWindowTimer()
    timer.start(TracingTimestamp.for_timestamp(timestamp_sec=16000, perf_counter=2000, validate_timestamp=False))

    with pytest.raises(OneLoggerError, match="Cannot stop timer with a stop time that is before the start time."):
        timer.stop(TracingTimestamp.for_timestamp(timestamp_sec=15000, perf_counter=1000, validate_timestamp=False))


@patch("time.time")
@patch("time.perf_counter")
def test_start_when_already_active(mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that starting an already active timer raises an error."""
    mock_time.return_value = 1000.0
    mock_perf_counter.return_value = 10.0
    timer = MultiWindowTimer()
    timer.start()
    with pytest.raises(OneLoggerError, match="Cannot start timer since it is already active"):
        timer.start()


@patch("time.time")
@patch("time.perf_counter")
def test_stop(mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that stopping the timer updates statistics correctly."""
    # Mock initial time for start
    mock_time.return_value = 1000.0
    mock_perf_counter.return_value = 10.0

    timer = MultiWindowTimer()
    timer.start()

    # Mock time for stop, simulating 5 seconds elapsed
    mock_time.return_value = 1005.0
    mock_perf_counter.return_value = 15.0

    timer.stop()
    assert timer.current_window_start is None
    assert timer.total_window_count == 1

    expected_duration = 5.0
    assert timer.total_time_sec == expected_duration
    assert timer.min_window_duration_sec == expected_duration
    assert timer.max_window_duration_sec == expected_duration
    assert timer.avg_window_duration_sec == expected_duration
    assert timer.latest_window_duration_sec == expected_duration
    assert timer.first_window_start == TracingTimestamp.for_timestamp(timestamp_sec=1000.0, perf_counter=10.0, validate_timestamp=False)
    assert timer.latest_window_end == TracingTimestamp.for_timestamp(timestamp_sec=1005.0, perf_counter=15.0, validate_timestamp=False)


def test_stop_when_not_active() -> None:  # No time mocking needed here
    """Test that stopping an inactive timer raises an error."""
    timer = MultiWindowTimer()
    with pytest.raises(OneLoggerError, match="Cannot stop timer since it is not active"):
        timer.stop()


@patch("time.time")
@patch("time.perf_counter")
def test_multiple_windows(mock_perf_counter: Mock, mock_time: Mock) -> None:
    """Test that multiple start/stop cycles update statistics correctly."""
    timer = MultiWindowTimer()

    # First window: 2 seconds duration
    mock_time.return_value = 1000.0
    mock_perf_counter.return_value = 10.0
    timer.start()
    mock_time.return_value = 1002.0
    mock_perf_counter.return_value = 12.0
    assert timer.current_window_start == TracingTimestamp.for_timestamp(timestamp_sec=1000.0, perf_counter=10.0, validate_timestamp=False)
    timer.stop()

    first_duration = 2.0
    assert not timer.current_window_start
    assert timer.total_window_count == 1
    assert timer.total_time_sec == first_duration
    assert timer.min_window_duration_sec == first_duration
    assert timer.max_window_duration_sec == first_duration
    assert timer.avg_window_duration_sec == first_duration
    assert timer.latest_window_duration_sec == first_duration

    # Second window: 3 seconds duration
    mock_time.return_value = 1005.0
    mock_perf_counter.return_value = 15.0
    timer.start()
    assert timer.current_window_start == TracingTimestamp.for_timestamp(timestamp_sec=1005.0, perf_counter=15.0, validate_timestamp=False)
    mock_time.return_value = 1008.0
    mock_perf_counter.return_value = 18.0
    timer.stop()

    second_duration = 3.0
    assert not timer.current_window_start
    assert timer.total_window_count == 2
    assert timer.total_time_sec == first_duration + second_duration
    assert timer.min_window_duration_sec == first_duration
    assert timer.max_window_duration_sec == second_duration
    assert timer.avg_window_duration_sec == (first_duration + second_duration) / 2
    assert timer.latest_window_duration_sec == second_duration

    # A long time passes ...
    mock_time.return_value = 2000.0
    mock_perf_counter.return_value = 15.0

    # Third window: 0.5 seconds duration
    timer.start()
    assert timer.current_window_start == TracingTimestamp.for_timestamp(timestamp_sec=2000.0, perf_counter=15.0, validate_timestamp=False)
    mock_time.return_value = 2000.05
    mock_perf_counter.return_value = 15.5
    timer.stop()

    third_duration = 0.5
    assert not timer.current_window_start
    assert timer.latest_window_duration_sec == third_duration
    assert timer.total_window_count == 3
    assert timer.total_time_sec == first_duration + second_duration + third_duration
    assert timer.min_window_duration_sec == third_duration
    assert timer.max_window_duration_sec == second_duration
    assert timer.avg_window_duration_sec == (first_duration + second_duration + third_duration) / 3
    assert timer.latest_window_duration_sec == third_duration

    assert timer.first_window_start == TracingTimestamp.for_timestamp(timestamp_sec=1000.0, perf_counter=10.0, validate_timestamp=False)
    assert timer.latest_window_end == TracingTimestamp.for_timestamp(timestamp_sec=2000.05, perf_counter=15.5, validate_timestamp=False)

    timer.reset()
    assert not timer.is_active
    assert not timer.current_window_start
    assert not timer.latest_window_end
    assert timer.total_window_count == 0
    assert timer.total_time_sec == 0.0
    assert timer.min_window_duration_sec == float("inf")
    assert timer.max_window_duration_sec == 0.0
    assert timer.avg_window_duration_sec == 0.0
    assert timer.latest_window_duration_sec == 0.0
    assert timer.first_window_start is None


def test_equality_same_timer() -> None:
    """Test that a timer is equal to itself."""
    timer = MultiWindowTimer()
    assert timer == timer


def test_equality_different_type() -> None:
    """Test that a timer is not equal to a different type."""
    timer = MultiWindowTimer()
    assert timer != "not a timer"
    assert timer != 42
    assert timer != None  # noqa: E711


def test_equality_empty_timers() -> None:
    """Test that two empty timers are equal."""
    timer1 = MultiWindowTimer()
    timer2 = MultiWindowTimer()
    assert timer1 == timer2


def test_equality_after_start() -> None:
    """Test that two timers are equal after starting them with the same timestamp."""
    with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
        mock_now.return_value = TracingTimestamp.for_timestamp(1000, 1.0, validate_timestamp=False)
        timer1 = MultiWindowTimer()
        timer2 = MultiWindowTimer()
        timer1.start()
        timer2.start()
        assert timer1 == timer2


def test_equality_after_stop() -> None:
    """Test that two timers are equal after starting and stopping them with the same timestamps."""
    with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
        # Start both timers
        mock_now.return_value = TracingTimestamp.for_timestamp(1000, 1.0, validate_timestamp=False)
        timer1 = MultiWindowTimer()
        timer2 = MultiWindowTimer()
        timer1.start()
        timer2.start()

        # Stop both timers
        mock_now.return_value = TracingTimestamp.for_timestamp(2000, 2.0, validate_timestamp=False)
        timer1.stop()
        timer2.stop()

        assert timer1 == timer2


def test_equality_different_timestamps() -> None:
    """Test that timers with different timestamps are not equal."""
    with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
        # Start first timer
        mock_now.return_value = TracingTimestamp.for_timestamp(1000, 1.0, validate_timestamp=False)
        timer1 = MultiWindowTimer()
        timer1.start()

        # Start second timer with different timestamp
        mock_now.return_value = TracingTimestamp.for_timestamp(2000, 2.0, validate_timestamp=False)
        timer2 = MultiWindowTimer()
        timer2.start()

        assert timer1 != timer2


def test_equality_after_reset() -> None:
    """Test that timers are equal after resetting them."""
    with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
        # Start and stop first timer
        mock_now.return_value = TracingTimestamp.for_timestamp(1000, 1.0, validate_timestamp=False)
        timer1 = MultiWindowTimer()
        timer1.start()
        mock_now.return_value = TracingTimestamp.for_timestamp(2000, 2.0, validate_timestamp=False)
        timer1.stop()

        # Start and stop second timer
        mock_now.return_value = TracingTimestamp.for_timestamp(3000, 3.0, validate_timestamp=False)
        timer2 = MultiWindowTimer()
        timer2.start()
        mock_now.return_value = TracingTimestamp.for_timestamp(4000, 4.0, validate_timestamp=False)
        timer2.stop()

        assert timer1 != timer2
        # Reset both timers
        timer1.reset()
        timer2.reset()

        assert timer1 == timer2


def test_hash_consistency() -> None:
    """Test that hash values are consistent for equal timers."""
    with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
        # Create two timers with same timestamps
        mock_now.return_value = TracingTimestamp.for_timestamp(1000, 1.0, validate_timestamp=False)
        timer1 = MultiWindowTimer()
        timer2 = MultiWindowTimer()
        timer1.start()
        timer2.start()

        mock_now.return_value = TracingTimestamp.for_timestamp(2000, 2.0, validate_timestamp=False)
        timer1.stop()
        timer2.stop()

        # Equal objects should have equal hashes
        assert timer1 == timer2
        assert hash(timer1) == hash(timer2)


def test_hash_different_timers() -> None:
    """Test that different timers have different hash values."""
    with patch("nv_one_logger.core.time.TracingTimestamp.now") as mock_now:
        # Create two timers with different timestamps
        mock_now.return_value = TracingTimestamp.for_timestamp(1000, 1.0, validate_timestamp=False)
        timer1 = MultiWindowTimer()
        timer1.start()

        mock_now.return_value = TracingTimestamp.for_timestamp(2000, 2.0, validate_timestamp=False)
        timer2 = MultiWindowTimer()
        timer2.start()

        # Different objects should have different hashes
        assert timer1 != timer2
        assert hash(timer1) != hash(timer2)


def test_hash_empty_timers() -> None:
    """Test that empty timers have the same hash value."""
    timer1 = MultiWindowTimer()
    timer2 = MultiWindowTimer()
    assert timer1 == timer2
    assert hash(timer1) == hash(timer2)
