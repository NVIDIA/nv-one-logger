# SPDX-License-Identifier: Apache-2.0
"""Tests for the time-related classes in one_logger.core."""

import time
from unittest.mock import Mock, patch

import pytest

from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.time import Timer, TracingTimestamp


class TestTracingTimestamp:
    """Tests for the TracingTimestamp class."""

    @patch("time.time")
    @patch("time.perf_counter")
    def test_now_creates_timestamp_with_current_time(self, mock_perf_counter: Mock, mock_time: Mock) -> None:
        """Verifies that now() creates a timestamp with the current time values."""
        mock_time.return_value = 123.456
        mock_perf_counter.return_value = 789.012

        timestamp = TracingTimestamp.now()

        assert timestamp.seconds_since_epoch == 123.456
        assert timestamp.perf_counter_seconds == 789.012

    def test_for_timestamp(self) -> None:
        """Verifies that for_timestamp() function with explicit perf_counter value."""
        ts_sec = time.time() - 10
        timestamp1 = TracingTimestamp.for_timestamp(timestamp_sec=ts_sec, perf_counter=800.1)

        assert timestamp1.seconds_since_epoch == ts_sec
        assert timestamp1.perf_counter_seconds == 800.1

    @patch("time.time")
    @patch("time.perf_counter")
    def test_for_timestamp_without_perf_counter(self, mock_perf_counter: Mock, mock_time: Mock) -> None:
        """Verifies that for_timestamp() called without a perf_counter can correctly calculate the perf_counter value."""
        mock_time.return_value = 200.0
        mock_perf_counter.return_value = 800.0

        timestamp1 = TracingTimestamp.for_timestamp(100.0)

        assert timestamp1.seconds_since_epoch == 100.0
        assert timestamp1.perf_counter_seconds == 700.0  # 800 - (200 - 100)

        # 500 seconds later
        mock_time.return_value = 700
        mock_perf_counter.return_value = 1300

        timestamp2 = TracingTimestamp.now()
        assert timestamp2.seconds_since_epoch - timestamp1.seconds_since_epoch == 600
        assert timestamp2.perf_counter_seconds - timestamp1.perf_counter_seconds == 600

    def test_from_json_creates_timestamp_from_dict(self) -> None:
        """Verifies that from_json() creates a timestamp from a dictionary."""
        data = {"seconds_since_epoch": 123.456, "perf_counter_seconds": 789.012}

        timestamp = TracingTimestamp.from_json(data)

        assert isinstance(timestamp, TracingTimestamp)
        assert timestamp.seconds_since_epoch == 123.456
        assert timestamp.perf_counter_seconds == 789.012

        assert timestamp.to_json() == data

    def test_for_timestamp_with_invalid_timestamp(self) -> None:
        """Verifies that for_timestamp() raises an exception if the timestamp is invalid."""
        with pytest.raises(OneLoggerError, match="timestamp is too old. This is meant to be the time in seconds since epoch."):
            TracingTimestamp.for_timestamp(1234)
        TracingTimestamp.for_timestamp(1234, validate_timestamp=False)

        with pytest.raises(OneLoggerError, match="timestamp is in the future. This is meant to be the time in seconds since epoch."):
            TracingTimestamp.for_timestamp(timestamp_sec=time.time() + 3000)
        TracingTimestamp.for_timestamp(timestamp_sec=time.time() + 3000, validate_timestamp=False)


class TestTimer:
    """Tests for the Timer class."""

    @patch("time.perf_counter")
    def test_start_sets_running_state(self, mock_perf_counter: Mock) -> None:
        """Verifies that start() sets the timer to running state."""
        mock_perf_counter.return_value = 100.0
        timer = Timer()
        timer.start()

        assert timer.running

    @patch("time.perf_counter")
    def test_stop_returns_elapsed_time(self, mock_perf_counter: Mock) -> None:
        """Verifies that stop() returns the correct elapsed time."""
        mock_perf_counter.return_value = 100.0
        timer = Timer()
        timer.start()
        mock_perf_counter.return_value += 50.0
        elapsed = timer.stop()

        assert elapsed == 50.0
        assert not timer.running

    @patch("time.perf_counter")
    def test_stop_with_reset_clears_elapsed_time(self, mock_perf_counter: Mock) -> None:
        """Verifies that stop(reset=True) clears the elapsed time."""
        mock_perf_counter.return_value = 100.0
        timer = Timer()
        timer.start()
        assert timer.running
        mock_perf_counter.return_value += 50.0
        assert timer.stop(reset=True) == 50.0
        assert not timer.running

        mock_perf_counter.return_value += 300.0
        timer.start()
        assert timer.running
        mock_perf_counter.return_value += 70
        assert timer.stop() == 70.0  # No accumulation
        assert not timer.running

    @patch("time.perf_counter")
    def test_multiple_start_stop_accumulates_time(self, mock_perf_counter: Mock) -> None:
        """Verifies that multiple start/stop cycles accumulate elapsed time."""
        timer = Timer()
        mock_perf_counter.return_value = 100.0
        timer.start()
        assert timer.running
        mock_perf_counter.return_value += 50.0
        assert timer.stop() == 50.0
        assert not timer.running

        mock_perf_counter.return_value += 300.0
        timer.start()
        assert timer.running
        mock_perf_counter.return_value += 70
        assert timer.stop() == 120.0  # 50 + 70
        assert not timer.running

    @patch("time.perf_counter")
    def test_reset_clears_state(self, mock_perf_counter: Mock) -> None:
        """Verifies that reset() clears all timer state."""
        timer = Timer()
        timer.start(TracingTimestamp.now())
        timer._elapsed_time_since_last_reset = 100.0

        timer.reset()

        assert not timer.running
        assert timer._elapsed_time_since_last_reset == 0.0
        assert timer._start_time is None

    @patch("time.perf_counter")
    def test_reset_with_start_starts_timer(self, mock_perf_counter: Mock) -> None:
        """Verifies that reset(start=True) starts the timer."""
        timer = Timer()
        mock_perf_counter.return_value = 100.0
        timer.start(TracingTimestamp.now())
        assert timer.running
        mock_perf_counter.return_value += 50.0
        assert timer.stop() == 50.0
        assert not timer.running
        timer.reset(start=True)

        assert timer.running
        mock_perf_counter.return_value += 30.0
        assert timer.stop() == 30.0
