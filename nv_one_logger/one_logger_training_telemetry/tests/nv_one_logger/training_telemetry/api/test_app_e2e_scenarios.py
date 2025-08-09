"""Unit tests for crash scenarios in nv-one-logger training telemetry.

These tests verify the behavior of nv-one-logger when training jobs crash with
active spans.
"""

import json
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
from nv_one_logger.core.event import StandardEventAttributeName
from nv_one_logger.core.time import TracingTimestamp
from nv_one_logger.exporter.file_exporter import FileExporter

from nv_one_logger.training_telemetry.api.callbacks import (
    on_app_end,
    on_app_start,
    on_load_checkpoint_start,
    on_train_end,
    on_train_start,
    on_training_single_iteration_end,
    on_training_single_iteration_start,
)
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.context import application, training_iteration, training_loop
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from .conftest import configure_provider_for_test
from .utils import advance_time

STARTING_PERF_COUNTER = 5000.0
STARTING_TIME = 120000.0


class TestCrashScenariosWithRealExporter:
    """Test crash scenarios using real JSON exporter to verify exported data."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self) -> None:
        """Set up and tear down the test environment with real file exporter."""
        # Create temporary file for JSON export
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.file_path = Path(self.temp_file.name)
        self.temp_file.close()

        # Create real file exporter
        self.exporter = FileExporter(file_path=self.file_path)
        self.exporter.initialize()

        yield

        # Cleanup
        self.exporter.close()
        if os.path.exists(self.file_path):
            os.unlink(self.file_path)

    @pytest.fixture(autouse=True)
    def configure_provider(self, config: TrainingTelemetryConfig, setup_teardown) -> None:
        """Configure the TrainingTelemetryProvider with real exporter."""
        configure_provider_for_test(config, self.exporter)

    @pytest.fixture(autouse=True)
    def initialize_time(self, mock_time: Mock, mock_perf_counter: Mock) -> None:
        """Initialize the time and perf counter mocks."""
        mock_time.return_value = STARTING_TIME
        mock_perf_counter.return_value = STARTING_PERF_COUNTER

    @pytest.fixture
    def training_recorder(self, config: TrainingTelemetryConfig) -> TrainingRecorder:
        """Create a TrainingRecorder instance for testing."""
        return TrainingTelemetryProvider.instance().recorder

    def _extract_exported_data(self) -> List[Dict[str, Any]]:
        """Extract and parse the exported JSON data from the file.

        Returns:
            List[Dict[str, Any]]: List of parsed JSON objects from the file.
        """
        # Close the file to ensure all data is written
        if self.exporter._file:
            self.exporter._file.close()

        with open(self.file_path, mode="r") as f:
            file_content = f.read().strip()
            if not file_content:
                return []

            lines = file_content.split("\n")
            # Filter out empty strings
            lines = [line for line in lines if line.strip()]
            return [json.loads(line) for line in lines]

    def _assert_span_exported(
        self,
        exported_data: List[Dict[str, Any]],
        span_name: str,
        record_type: str = "start",
    ) -> Dict[str, Any]:
        """Assert that a span with the given name was exported with the given record type.

        Args:
            exported_data: List of exported JSON records
            span_name: Expected span name
            record_type: Expected record type ("start", "stop", "complete")

        Returns:
            The exported span record
        """
        matching_records = [record for record in exported_data if record.get("type") == record_type and record.get("name") == span_name]
        assert len(matching_records) > 0, f"Expected span '{span_name}' with type '{record_type}' " "not found in exported data"
        return matching_records[0]

    def _assert_telemetry_data_error_exported(self, exported_data: List[Dict[str, Any]], expected_message: str) -> Dict[str, Any]:
        """Assert that a telemetry data error was exported with the expected message.

        Args:
            exported_data: List of exported JSON records
            expected_message: Expected error message

        Returns:
            The exported telemetry data error record
        """
        error_records = [record for record in exported_data if record.get("type") == "telemetry_data_error"]
        assert len(error_records) > 0, "Expected telemetry data error not found in exported data"

        error_record = error_records[0]
        error_event = error_record.get("error", {})
        actual_message = error_event.get("attributes", {}).get(StandardEventAttributeName.ERROR_MESSAGE, "")

        assert expected_message in actual_message, f"Expected error message '{expected_message}' " f"not found in '{actual_message}'"
        return error_record

    def test_context_manager_api_crash_with_exception(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test context manager API crash behavior with exception using real JSON exporter."""
        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        try:
            with application():
                with training_loop(train_iterations_start=0, train_iterations_target_or_fn=10):
                    for iteration in range(1, 4):  # Will crash at iteration 3
                        with training_iteration():
                            advance_time(mock_time, mock_perf_counter, 0.1)

                            if iteration == 3:
                                # Simulate crash
                                raise RuntimeError(f"Simulated crash at iteration {iteration}")

                            advance_time(mock_time, mock_perf_counter, 0.1)
        except RuntimeError:
            # Exception expected
            pass

        # Verify that context managers cleaned up properly
        assert len(training_recorder._spans) == 0
        assert training_recorder._closed

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application and training_loop spans
        # (training_iteration not exported due to blacklist)
        app_span_record = self._assert_span_exported(exported_data, "application", "start")
        training_loop_record = self._assert_span_exported(exported_data, "training_loop", "complete")

        # Should have stop events for application span
        app_stop_record = self._assert_span_exported(exported_data, "application", "stop")

        # Verify span IDs match
        assert app_span_record["id"] == app_stop_record["id"]

        # Verify training_loop complete record has both start and stop events
        assert "start_event" in training_loop_record
        assert "stop_event" in training_loop_record

        # Verify no training_iteration spans were exported (due to blacklist)
        training_iteration_records = [record for record in exported_data if record.get("name") == "training_single_iteration"]
        assert len(training_iteration_records) == 0, "Training iteration spans should not be exported due to blacklist"

    def test_signal_based_crash_cleanup(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test signal-based crash cleanup behavior with real JSON exporter."""
        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=10,
            train_samples_target_or_fn=320,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Start a training iteration
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1)

        # Simulate signal handler cleanup
        # Report telemetry data error
        from nv_one_logger.core.attributes import Attribute, Attributes

        attributes = Attributes()
        attributes.add_attribute(Attribute("signal", signal.SIGTERM))
        attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))

        training_recorder.telemetry_data_error("Application terminated by signal", attributes=attributes)

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application and training_loop spans
        self._assert_span_exported(exported_data, "application", "start")
        # Note: training_loop span is not exported because it's still active when telemetry data error is reported

        # Should have exported telemetry data error
        error_record = self._assert_telemetry_data_error_exported(exported_data, "Application terminated by signal")

        # Verify error attributes
        error_event = error_record.get("error", {})
        error_attributes = error_event.get("attributes", {})
        assert error_attributes.get("signal") == signal.SIGTERM
        assert error_attributes.get("active_spans_count") == 3  # app, training_loop, training_iteration

    def test_manual_cleanup_after_crash(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test manual cleanup after a crash scenario with real JSON exporter."""
        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Step 1: Simulate crash scenario
        app_span = on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        training_loop_span = on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=10,
            train_samples_target_or_fn=320,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        training_iteration_span = on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1)

        # Verify we have active spans
        assert len(training_recorder._spans) == 3

        # Step 2: Perform manual cleanup (what should happen in a proper shutdown)
        advance_time(mock_time, mock_perf_counter, 0.1)

        # Stop spans in reverse order (as done in DefaultRecorder.close())
        training_recorder.stop(training_iteration_span, stop_time=TracingTimestamp.now())
        training_recorder.stop(training_loop_span, stop_time=TracingTimestamp.now())
        training_recorder.stop(app_span, stop_time=TracingTimestamp.now())

        # Step 3: Close the recorder
        training_recorder.close()
        assert training_recorder._closed

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported all spans
        self._assert_span_exported(exported_data, "application", "start")
        self._assert_span_exported(exported_data, "application", "stop")
        self._assert_span_exported(exported_data, "training_loop", "complete")

        # Training iteration should not be exported due to blacklist
        training_iteration_records = [record for record in exported_data if record.get("name") == "training_single_iteration"]
        assert len(training_iteration_records) == 0, "Training iteration spans should not be exported due to blacklist"

    def test_multi_window_timer_statistics(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that multi-window-timer statistics are tracked correctly."""
        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=10,
            train_samples_target_or_fn=320,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        for i in range(5):
            on_training_single_iteration_start()
            advance_time(mock_time, mock_perf_counter, 0.1 + i * 0.01)
            on_training_single_iteration_end()
            advance_time(mock_time, mock_perf_counter, 0.02)

        # Verify that the multi-window timer has accumulated statistics
        training_iteration_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        assert training_iteration_timer.total_window_count == 5, "Timer should have tracked 5 training iterations"
        assert training_iteration_timer.total_time_sec > 0, "Timer should have accumulated total time"
        assert training_iteration_timer.min_window_duration_sec > 0, "Timer should have minimum duration"
        assert training_iteration_timer.max_window_duration_sec > 0, "Timer should have maximum duration"
        assert training_iteration_timer.avg_window_duration_sec > 0, "Timer should have average duration"

        # Simulate crash and report telemetry data error
        from nv_one_logger.core.attributes import Attribute, Attributes

        attributes = Attributes()
        attributes.add_attribute(Attribute("crash_reason", "simulated_crash"))
        attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))
        attributes.add_attribute(Attribute("crash_mode", "multi_window_timer_test"))

        training_recorder.telemetry_data_error("Crash after several training iterations", attributes=attributes)

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application start
        self._assert_span_exported(exported_data, "application", "start")

        # Should have exported telemetry data error
        error_record = self._assert_telemetry_data_error_exported(exported_data, "Crash after several training iterations")
        error_event = error_record.get("error", {})
        error_attributes = error_event.get("attributes", {})
        assert error_attributes.get("crash_reason") == "simulated_crash"
        assert error_attributes.get("active_spans_count") == 2, "Expected 2 active spans (app, training_loop)"
        assert error_attributes.get("crash_mode") == "multi_window_timer_test"

        # Note: Multi-window timer statistics are not exported in the current
        # implementation.
        # The timers are used internally for tracking but their statistics are
        # not automatically exported.
        # This test verifies that the timers work correctly and accumulate
        # statistics as expected.
        training_metrics_events = [
            record for record in exported_data if (record.get("type") == "event" and record.get("event", {}).get("name") == "training_metrics_update")
        ]
        assert len(training_metrics_events) == 0, "Expected 0 training metrics events, got " f"{len(training_metrics_events)}"

    def test_crash_with_telemetry_data_error_reporting(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test crash scenario with proper telemetry data error reporting using real JSON exporter."""
        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=10,
            train_samples_target_or_fn=320,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1)

        # Simulate crash
        # Report telemetry data error with detailed attributes
        from nv_one_logger.core.attributes import Attribute, Attributes

        attributes = Attributes()
        attributes.add_attribute(Attribute("crash_reason", "simulated_crash"))
        attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))
        attributes.add_attribute(Attribute("crash_iteration", 3))
        attributes.add_attribute(Attribute("crash_mode", "exception"))

        training_recorder.telemetry_data_error(
            "Application crashed with incomplete telemetry data",
            attributes=attributes,
        )

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported spans
        self._assert_span_exported(exported_data, "application", "start")
        # Note: training_loop span is not exported because it's still active when telemetry data error is reported

        # Should have exported telemetry data error with detailed attributes
        error_record = self._assert_telemetry_data_error_exported(exported_data, "Application crashed with incomplete telemetry data")

        # Verify detailed error attributes
        error_event = error_record.get("error", {})
        error_attributes = error_event.get("attributes", {})
        assert error_attributes.get("crash_reason") == "simulated_crash"
        assert error_attributes.get("active_spans_count") == 3
        assert error_attributes.get("crash_iteration") == 3
        assert error_attributes.get("crash_mode") == "exception"

    def test_crash_recovery_scenario(
        self,
        training_recorder: TrainingRecorder,
        config: TrainingTelemetryConfig,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test recovery scenario after a crash using real JSON exporter."""
        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # First: Simulate a crash
        try:
            with application():
                with training_loop(train_iterations_start=0, train_iterations_target_or_fn=10):
                    with training_iteration():
                        advance_time(mock_time, mock_perf_counter, 0.1)  # Ensure sufficient time
                        advance_time(mock_time, mock_perf_counter, 0.1)  # Ensure sufficient time
                        raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass

        # Verify crash state
        assert len(training_recorder._spans) == 0
        assert training_recorder._closed

        # Get exported data from first crash
        first_crash_data = self._extract_exported_data()

        # Should have exported spans from first crash
        self._assert_span_exported(first_crash_data, "application", "start")
        self._assert_span_exported(first_crash_data, "application", "stop")
        self._assert_span_exported(first_crash_data, "training_loop", "complete")

        # Second: Simulate recovery and successful completion
        # Create new exporter for recovery scenario
        recovery_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        recovery_file_path = Path(recovery_temp_file.name)
        recovery_temp_file.close()

        try:
            recovery_exporter = FileExporter(file_path=recovery_file_path)
            recovery_exporter.initialize()

            # Configure provider with new exporter
            configure_provider_for_test(config, recovery_exporter)

            with application():
                with training_loop(train_iterations_start=0, train_iterations_target_or_fn=5):
                    for _ in range(1, 6):
                        with training_iteration():
                            advance_time(mock_time, mock_perf_counter, 0.1)
                            advance_time(mock_time, mock_perf_counter, 0.1)

            # Verify successful completion
            recovery_recorder = TrainingTelemetryProvider.instance().recorder
            assert len(recovery_recorder._spans) == 0
            assert recovery_recorder._closed

            # Verify recovery exported data
            recovery_exporter.close()
            with open(recovery_file_path, mode="r") as f:
                recovery_content = f.read().strip()
                recovery_data = [json.loads(line) for line in recovery_content.split("\n") if line.strip()]

            # Should have exported spans from recovery
            self._assert_span_exported(recovery_data, "application", "start")
            self._assert_span_exported(recovery_data, "application", "stop")
            self._assert_span_exported(recovery_data, "training_loop", "complete")

        finally:
            # Cleanup recovery file
            if os.path.exists(recovery_file_path):
                os.unlink(recovery_file_path)

    def test_timer_data_export_with_multiple_metrics_updates(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when there are multiple metrics updates."""
        # Create a custom config with smaller log_every_n_train_iterations for testing
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,  # Smaller value to trigger more frequent exports
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        training_recorder = TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Simulate many training iterations to trigger metrics export
        # The config has log_every_n_train_iterations=5, so metrics will be exported at iterations 5, 10, 15, etc.
        for i in range(15):  # Do 15 iterations to get multiple exports
            on_training_single_iteration_start()
            advance_time(mock_time, mock_perf_counter, 0.1 + i * 0.01)  # Varying iteration times
            on_training_single_iteration_end()
            advance_time(mock_time, mock_perf_counter, 0.02)

        # End training loop properly to ensure all data is exported
        on_train_end()
        advance_time(mock_time, mock_perf_counter, 0.1)

        # End application properly
        on_app_end()
        advance_time(mock_time, mock_perf_counter, 0.1)

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application spans
        self._assert_span_exported(exported_data, "application", "start")
        self._assert_span_exported(exported_data, "application", "stop")

        # Should have exported training_loop spans
        self._assert_span_exported(exported_data, "training_loop", "start")
        self._assert_span_exported(exported_data, "training_loop", "stop")

        # Look for training metrics update events (which contain timer data)
        training_metrics_events = [
            record for record in exported_data if record.get("type") == "event" and record.get("event", {}).get("name") == "training_metrics_update"
        ]
        assert len(training_metrics_events) == 3, f"Expected 3 training metrics events, got {len(training_metrics_events)}"

        # Validate the first training metrics event (should be around iteration 5)
        first_metrics_event = training_metrics_events[0]
        event_data = first_metrics_event.get("event", {})
        attributes = event_data.get("attributes", {})

        # Validate timer-related attributes
        assert "num_iterations" in attributes, "Training metrics should include num_iterations"
        assert "avg_iteration_time_sec" in attributes, "Training metrics should include avg_iteration_time_sec"
        assert "min_iteration_time_sec" in attributes, "Training metrics should include min_iteration_time_sec"
        assert "max_iteration_time_sec" in attributes, "Training metrics should include max_iteration_time_sec"
        assert "total_iteration_time_sec" in attributes, "Training metrics should include total_iteration_time_sec"

        # Validate timer data values
        num_iterations = attributes.get("num_iterations")
        assert num_iterations == 5, f"First metrics event should have 5 iterations, got {num_iterations}"

        avg_time = attributes.get("avg_iteration_time_sec")
        min_time = attributes.get("min_iteration_time_sec")
        max_time = attributes.get("max_iteration_time_sec")
        total_time = attributes.get("total_iteration_time_sec")

        assert avg_time > 0, f"Average iteration time should be positive, got {avg_time}"
        assert min_time > 0, f"Minimum iteration time should be positive, got {min_time}"
        assert max_time > 0, f"Maximum iteration time should be positive, got {max_time}"
        assert total_time > 0, f"Total iteration time should be positive, got {total_time}"
        assert min_time <= avg_time <= max_time, f"Timer statistics should be consistent: min={min_time}, avg={avg_time}, max={max_time}"
        assert (
            abs(total_time - (avg_time * num_iterations)) < 0.001
        ), f"Total time should equal avg * num_iterations: {total_time} vs {avg_time * num_iterations}"

        # Validate the second training metrics event (should be around iteration 10)
        if len(training_metrics_events) >= 2:
            second_metrics_event = training_metrics_events[1]
            event_data = second_metrics_event.get("event", {})
            attributes = event_data.get("attributes", {})

            num_iterations = attributes.get("num_iterations")
            assert num_iterations == 10, f"Second metrics event should have 10 iterations, got {num_iterations}"

            # The second event should have accumulated more time
            total_time_second = attributes.get("total_iteration_time_sec")
            assert total_time_second > total_time, f"Second event should have more total time: {total_time_second} > {total_time}"

        # Verify that the multi-window timer internal state is correct
        training_iteration_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        assert training_iteration_timer.total_window_count == 15, "Timer should have tracked 15 training iterations"
        assert training_iteration_timer.total_time_sec > 0, "Timer should have accumulated total time"
        assert training_iteration_timer.min_window_duration_sec > 0, "Timer should have minimum duration"
        assert training_iteration_timer.max_window_duration_sec > 0, "Timer should have maximum duration"
        assert training_iteration_timer.avg_window_duration_sec > 0, "Timer should have average duration"

        # Verify that the timer statistics in the last exported event match the internal timer state
        last_metrics_event = training_metrics_events[-1]
        event_data = last_metrics_event.get("event", {})
        attributes = event_data.get("attributes", {})

        exported_total_time = attributes.get("total_iteration_time_sec")
        exported_avg_time = attributes.get("avg_iteration_time_sec")
        exported_min_time = attributes.get("min_iteration_time_sec")
        exported_max_time = attributes.get("max_iteration_time_sec")

        # Allow for small floating point differences
        assert (
            abs(exported_total_time - training_iteration_timer.total_time_sec) < 0.001
        ), f"Exported total time should match timer state: {exported_total_time} vs {training_iteration_timer.total_time_sec}"
        assert (
            abs(exported_avg_time - training_iteration_timer.avg_window_duration_sec) < 0.001
        ), f"Exported avg time should match timer state: {exported_avg_time} vs {training_iteration_timer.avg_window_duration_sec}"
        assert (
            abs(exported_min_time - training_iteration_timer.min_window_duration_sec) < 0.001
        ), f"Exported min time should match timer state: {exported_min_time} vs {training_iteration_timer.min_window_duration_sec}"
        assert (
            abs(exported_max_time - training_iteration_timer.max_window_duration_sec) < 0.001
        ), f"Exported max time should match timer state: {exported_max_time} vs {training_iteration_timer.max_window_duration_sec}"

    def test_timer_data_export_with_crash_at_iteration_after_the_first_log_interval(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when the application crashes at iteration 7 after the first log interval."""
        # Create a custom config with smaller log_every_n_train_iterations for testing
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,  # Smaller value to trigger more frequent exports
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        training_recorder = TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Simulate training iterations until crash at iteration 7
        # The config has log_every_n_train_iterations=5, so metrics will be exported at iteration 5
        # We'll crash at iteration 7, so we should get one metrics export at iteration 5
        for i in range(7):  # Do 7 iterations, crash at iteration 7
            on_training_single_iteration_start()
            advance_time(mock_time, mock_perf_counter, 0.1 + i * 0.01)  # Varying iteration times
            on_training_single_iteration_end()
            advance_time(mock_time, mock_perf_counter, 0.02)

        # Simulate crash at iteration 7 by reporting telemetry data error
        from nv_one_logger.core.attributes import Attribute, Attributes

        attributes = Attributes()
        attributes.add_attribute(Attribute("crash_reason", "simulated_crash_at_iteration_7"))
        attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))
        attributes.add_attribute(Attribute("crash_iteration", 7))
        attributes.add_attribute(Attribute("crash_mode", "timer_data_export_test"))

        training_recorder.telemetry_data_error(
            "Application crashed at iteration 7 during timer data export test",
            attributes=attributes,
        )

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application spans
        self._assert_span_exported(exported_data, "application", "start")
        # Note: application stop is not exported because it's still active when telemetry data error is reported

        # Should have exported training_loop spans
        self._assert_span_exported(exported_data, "training_loop", "start")
        # Note: training_loop stop is not exported because it's still active when telemetry data error is reported

        # Look for training metrics update events (which contain timer data)
        training_metrics_events = [
            record for record in exported_data if record.get("type") == "event" and record.get("event", {}).get("name") == "training_metrics_update"
        ]
        assert len(training_metrics_events) == 1, f"Expected 1 training metrics event (at iteration 5), got {len(training_metrics_events)}"

        # Validate the training metrics event (should be at iteration 5)
        metrics_event = training_metrics_events[0]
        event_data = metrics_event.get("event", {})
        attributes = event_data.get("attributes", {})

        # Validate timer-related attributes
        assert "num_iterations" in attributes, "Training metrics should include num_iterations"
        assert "avg_iteration_time_sec" in attributes, "Training metrics should include avg_iteration_time_sec"
        assert "min_iteration_time_sec" in attributes, "Training metrics should include min_iteration_time_sec"
        assert "max_iteration_time_sec" in attributes, "Training metrics should include max_iteration_time_sec"
        assert "total_iteration_time_sec" in attributes, "Training metrics should include total_iteration_time_sec"

        # Validate timer data values
        num_iterations = attributes.get("num_iterations")
        assert num_iterations == 5, f"Metrics event should have 5 iterations, got {num_iterations}"

        avg_time = attributes.get("avg_iteration_time_sec")
        min_time = attributes.get("min_iteration_time_sec")
        max_time = attributes.get("max_iteration_time_sec")
        total_time = attributes.get("total_iteration_time_sec")

        assert avg_time > 0, f"Average iteration time should be positive, got {avg_time}"
        assert min_time > 0, f"Minimum iteration time should be positive, got {min_time}"
        assert max_time > 0, f"Maximum iteration time should be positive, got {max_time}"
        assert total_time > 0, f"Total iteration time should be positive, got {total_time}"
        assert min_time <= avg_time <= max_time, f"Timer statistics should be consistent: min={min_time}, avg={avg_time}, max={max_time}"
        assert (
            abs(total_time - (avg_time * num_iterations)) < 0.001
        ), f"Total time should equal avg * num_iterations: {total_time} vs {avg_time * num_iterations}"

        # Verify that the multi-window timer internal state is correct
        training_iteration_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        assert training_iteration_timer.total_window_count == 7, "Timer should have tracked 7 training iterations"
        assert training_iteration_timer.total_time_sec > 0, "Timer should have accumulated total time"
        assert training_iteration_timer.min_window_duration_sec > 0, "Timer should have minimum duration"
        assert training_iteration_timer.max_window_duration_sec > 0, "Timer should have maximum duration"
        assert training_iteration_timer.avg_window_duration_sec > 0, "Timer should have average duration"

        # Verify that the timer statistics in the exported event match the internal timer state
        # Note: The exported event only contains data for the first 5 iterations (log_every_n_train_iterations=5)
        # but the internal timer has data for all 7 iterations
        exported_total_time = attributes.get("total_iteration_time_sec")
        exported_avg_time = attributes.get("avg_iteration_time_sec")
        exported_min_time = attributes.get("min_iteration_time_sec")
        exported_max_time = attributes.get("max_iteration_time_sec")

        # The exported event should contain data for the first 5 iterations only
        # The internal timer has data for all 7 iterations, so we need to calculate what the first 5 iterations should be
        # Since we're using varying iteration times (0.1 + i * 0.01), the first 5 iterations should have:
        # iteration 0: 0.1, iteration 1: 0.11, iteration 2: 0.12, iteration 3: 0.13, iteration 4: 0.14
        expected_first_5_total_time = 0.1 + 0.11 + 0.12 + 0.13 + 0.14  # 0.6 seconds
        expected_first_5_avg_time = expected_first_5_total_time / 5  # 0.12 seconds
        expected_first_5_min_time = 0.1  # iteration 0
        expected_first_5_max_time = 0.14  # iteration 4

        # Allow for small floating point differences
        assert (
            abs(exported_total_time - expected_first_5_total_time) < 0.001
        ), f"Exported total time should match expected first 5 iterations: {exported_total_time} vs {expected_first_5_total_time}"
        assert (
            abs(exported_avg_time - expected_first_5_avg_time) < 0.001
        ), f"Exported avg time should match expected first 5 iterations: {exported_avg_time} vs {expected_first_5_avg_time}"
        assert (
            abs(exported_min_time - expected_first_5_min_time) < 0.001
        ), f"Exported min time should match expected first 5 iterations: {exported_min_time} vs {expected_first_5_min_time}"
        assert (
            abs(exported_max_time - expected_first_5_max_time) < 0.001
        ), f"Exported max time should match expected first 5 iterations: {exported_max_time} vs {expected_first_5_max_time}"

        # Verify that the internal timer state reflects all 7 iterations
        # The internal timer should have accumulated time for all 7 iterations
        expected_all_7_total_time = 0.1 + 0.11 + 0.12 + 0.13 + 0.14 + 0.15 + 0.16  # 0.91 seconds
        expected_all_7_avg_time = expected_all_7_total_time / 7  # ~0.13 seconds
        expected_all_7_min_time = 0.1  # iteration 0
        expected_all_7_max_time = 0.16  # iteration 6

        assert (
            abs(training_iteration_timer.total_time_sec - expected_all_7_total_time) < 0.001
        ), f"Internal timer total time should match all 7 iterations: {training_iteration_timer.total_time_sec} vs {expected_all_7_total_time}"
        assert (
            abs(training_iteration_timer.avg_window_duration_sec - expected_all_7_avg_time) < 0.001
        ), f"Internal timer avg time should match all 7 iterations: {training_iteration_timer.avg_window_duration_sec} vs {expected_all_7_avg_time}"
        assert (
            abs(training_iteration_timer.min_window_duration_sec - expected_all_7_min_time) < 0.001
        ), f"Internal timer min time should match all 7 iterations: {training_iteration_timer.min_window_duration_sec} vs {expected_all_7_min_time}"
        assert (
            abs(training_iteration_timer.max_window_duration_sec - expected_all_7_max_time) < 0.001
        ), f"Internal timer max time should match all 7 iterations: {training_iteration_timer.max_window_duration_sec} vs {expected_all_7_max_time}"

        # Should have exported telemetry data error
        error_record = self._assert_telemetry_data_error_exported(
            exported_data,
            "Application crashed at iteration 7 during timer data export test",
        )
        error_event = error_record.get("error", {})
        error_attributes = error_event.get("attributes", {})
        assert error_attributes.get("crash_reason") == "simulated_crash_at_iteration_7"
        assert error_attributes.get("active_spans_count") == 2  # app, training_loop
        assert error_attributes.get("crash_iteration") == 7
        assert error_attributes.get("crash_mode") == "timer_data_export_test"

    def test_timer_data_export_with_crash_before_iteration_end_after_the_first_log_interval(  # noqa: E501
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when crashes."""
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,  # Smaller value to trigger more frequent exports
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        training_recorder = TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Simulate training iterations until crash at iteration 7
        # The config has log_every_n_train_iterations=5, so metrics will be exported at iteration 5
        # We'll crash at iteration 7 before calling on_training_single_iteration_end
        for i in range(6):  # Do 6 complete iterations (0-5)
            on_training_single_iteration_start()
            advance_time(mock_time, mock_perf_counter, 0.1 + i * 0.01)  # Varying iteration times
            on_training_single_iteration_end()
            advance_time(mock_time, mock_perf_counter, 0.02)

        # Start iteration 7 but crash before ending it
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1 + 6 * 0.01)  # iteration 6 time (0.16)

        # Simulate crash before calling on_training_single_iteration_end
        from nv_one_logger.core.attributes import Attribute, Attributes

        attributes = Attributes()
        attributes.add_attribute(Attribute("crash_reason", "simulated_crash_before_iteration_end"))
        attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))
        attributes.add_attribute(Attribute("crash_iteration", 7))
        attributes.add_attribute(Attribute("crash_mode", "timer_data_export_test_before_end"))

        training_recorder.telemetry_data_error(
            "Application crashed before on_training_single_iteration_end at iteration 7",
            attributes=attributes,
        )

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application spans
        self._assert_span_exported(exported_data, "application", "start")
        # Note: application stop is not exported because it's still active when telemetry data error is reported

        # Should have exported training_loop spans
        self._assert_span_exported(exported_data, "training_loop", "start")
        # Note: training_loop stop is not exported because it's still active when telemetry data error is reported

        # Note: training_single_iteration spans are not exported due to blacklist, even when incomplete
        training_iteration_records = [record for record in exported_data if record.get("name") == "training_single_iteration"]
        assert len(training_iteration_records) == 0, "Training iteration spans should not be exported due to blacklist"

        # Look for training metrics update events (which contain timer data)
        training_metrics_events = [
            record for record in exported_data if record.get("type") == "event" and record.get("event", {}).get("name") == "training_metrics_update"
        ]
        assert len(training_metrics_events) == 1, f"Expected 1 training metrics event (at iteration 5), got {len(training_metrics_events)}"

        # Validate the training metrics event (should be at iteration 5)
        metrics_event = training_metrics_events[0]
        event_data = metrics_event.get("event", {})
        attributes = event_data.get("attributes", {})

        # Validate timer-related attributes
        assert "num_iterations" in attributes, "Training metrics should include num_iterations"
        assert "avg_iteration_time_sec" in attributes, "Training metrics should include avg_iteration_time_sec"
        assert "min_iteration_time_sec" in attributes, "Training metrics should include min_iteration_time_sec"
        assert "max_iteration_time_sec" in attributes, "Training metrics should include max_iteration_time_sec"
        assert "total_iteration_time_sec" in attributes, "Training metrics should include total_iteration_time_sec"

        # Validate timer data values
        num_iterations = attributes.get("num_iterations")
        assert num_iterations == 5, f"Metrics event should have 5 iterations, got {num_iterations}"

        avg_time = attributes.get("avg_iteration_time_sec")
        min_time = attributes.get("min_iteration_time_sec")
        max_time = attributes.get("max_iteration_time_sec")
        total_time = attributes.get("total_iteration_time_sec")

        assert avg_time > 0, f"Average iteration time should be positive, got {avg_time}"
        assert min_time > 0, f"Minimum iteration time should be positive, got {min_time}"
        assert max_time > 0, f"Maximum iteration time should be positive, got {max_time}"
        assert total_time > 0, f"Total iteration time should be positive, got {total_time}"
        assert min_time <= avg_time <= max_time, f"Timer statistics should be consistent: min={min_time}, avg={avg_time}, max={max_time}"
        assert (
            abs(total_time - (avg_time * num_iterations)) < 0.001
        ), f"Total time should equal avg * num_iterations: {total_time} vs {avg_time * num_iterations}"

        # Verify that the multi-window timer internal state is correct
        training_iteration_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        assert training_iteration_timer.total_window_count == 7, "Timer should have tracked 7 training iterations (6 completed + 1 started but not finished)"
        assert training_iteration_timer.total_time_sec > 0, "Timer should have accumulated total time"
        assert training_iteration_timer.min_window_duration_sec > 0, "Timer should have minimum duration"
        assert training_iteration_timer.max_window_duration_sec > 0, "Timer should have maximum duration"
        assert training_iteration_timer.avg_window_duration_sec > 0, "Timer should have average duration"

        # Verify that the timer statistics in the exported event match the internal timer state
        # Note: The exported event only contains data for the first 5 iterations (log_every_n_train_iterations=5)
        # The internal timer has data for 6 completed iterations, so we need to calculate what the first 5 iterations should be
        # Since we're using varying iteration times (0.1 + i * 0.01), the first 5 iterations should have:
        # iteration 0: 0.1, iteration 1: 0.11, iteration 2: 0.12, iteration 3: 0.13, iteration 4: 0.14
        expected_first_5_total_time = 0.1 + 0.11 + 0.12 + 0.13 + 0.14  # 0.6 seconds
        expected_first_5_avg_time = expected_first_5_total_time / 5  # 0.12 seconds
        expected_first_5_min_time = 0.1  # iteration 0
        expected_first_5_max_time = 0.14  # iteration 4

        # Allow for small floating point differences
        assert (
            abs(total_time - expected_first_5_total_time) < 0.001
        ), f"Exported total time should match expected first 5 iterations: {total_time} vs {expected_first_5_total_time}"
        assert (
            abs(avg_time - expected_first_5_avg_time) < 0.001
        ), f"Exported avg time should match expected first 5 iterations: {avg_time} vs {expected_first_5_avg_time}"
        assert (
            abs(min_time - expected_first_5_min_time) < 0.001
        ), f"Exported min time should match expected first 5 iterations: {min_time} vs {expected_first_5_min_time}"
        assert (
            abs(max_time - expected_first_5_max_time) < 0.001
        ), f"Exported max time should match expected first 5 iterations: {max_time} vs {expected_first_5_max_time}"

        # Verify that the internal timer state reflects all 6 completed iterations
        # The internal timer should have accumulated time for all 6 completed iterations
        # Note: The 7th iteration was started but not completed, so it doesn't contribute to the statistics
        expected_all_6_total_time = 0.1 + 0.11 + 0.12 + 0.13 + 0.14 + 0.15  # 0.75 seconds
        expected_all_6_avg_time = expected_all_6_total_time / 6  # 0.125 seconds
        expected_all_6_min_time = 0.1  # iteration 0
        expected_all_6_max_time = 0.15  # iteration 5

        assert (
            abs(training_iteration_timer.total_time_sec - expected_all_6_total_time) < 0.001
        ), f"Internal timer total time should match all 6 completed iterations: {training_iteration_timer.total_time_sec} vs {expected_all_6_total_time}"
        assert (
            abs(training_iteration_timer.avg_window_duration_sec - expected_all_6_avg_time) < 0.001
        ), f"Internal timer avg time should match all 6 completed iterations: {training_iteration_timer.avg_window_duration_sec} vs {expected_all_6_avg_time}"
        assert (
            abs(training_iteration_timer.min_window_duration_sec - expected_all_6_min_time) < 0.001
        ), f"Internal timer min time should match all 6 completed iterations: {training_iteration_timer.min_window_duration_sec} vs {expected_all_6_min_time}"
        assert (
            abs(training_iteration_timer.max_window_duration_sec - expected_all_6_max_time) < 0.001
        ), f"Internal timer max time should match all 6 completed iterations: {training_iteration_timer.max_window_duration_sec} vs {expected_all_6_max_time}"

        # Verify that the timer is still active for the incomplete iteration 7
        assert training_iteration_timer.is_active, "Timer should still be active for the incomplete iteration 7"

        # Should have exported telemetry data error
        error_record = self._assert_telemetry_data_error_exported(
            exported_data,
            "Application crashed before on_training_single_iteration_end at iteration 7",
        )
        error_event = error_record.get("error", {})
        error_attributes = error_event.get("attributes", {})
        assert error_attributes.get("crash_reason") == "simulated_crash_before_iteration_end"
        assert error_attributes.get("active_spans_count") == 3  # app, training_loop, training_single_iteration
        assert error_attributes.get("crash_iteration") == 7
        assert error_attributes.get("crash_mode") == "timer_data_export_test_before_end"

    def test_timer_data_export_with_crash_and_app_end_failure_handler(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when the application crashes and calls on_app_end as a failure handler."""
        # Create a custom config with smaller log_every_n_train_iterations for testing
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,  # Smaller value to trigger more frequent exports
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        training_recorder = TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Simulate training iterations until crash at iteration 7
        # The config has log_every_n_train_iterations=5, so metrics will be exported at iteration 5
        # We'll crash at iteration 7 before calling on_training_single_iteration_end
        for i in range(6):  # Do 6 complete iterations (0-5)
            on_training_single_iteration_start()
            advance_time(mock_time, mock_perf_counter, 0.1 + i * 0.01)  # Varying iteration times
            on_training_single_iteration_end()
            advance_time(mock_time, mock_perf_counter, 0.02)

        # Start iteration 7 but crash before ending it
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1 + 6 * 0.01)  # iteration 6 time (0.16)

        # Simulate crash before calling on_training_single_iteration_end
        from nv_one_logger.core.attributes import Attribute, Attributes

        attributes = Attributes()
        attributes.add_attribute(Attribute("crash_reason", "simulated_crash_with_app_end_handler"))
        attributes.add_attribute(Attribute("active_spans_count", len(training_recorder._spans)))
        attributes.add_attribute(Attribute("crash_iteration", 7))
        attributes.add_attribute(Attribute("crash_mode", "timer_data_export_test_with_app_end"))

        training_recorder.telemetry_data_error(
            "Application crashed before on_training_single_iteration_end at iteration 7",
            attributes=attributes,
        )

        # Simulate calling on_app_end as a failure handler
        advance_time(mock_time, mock_perf_counter, 0.1)
        on_app_end()

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application spans (both start and stop since on_app_end succeeded)
        self._assert_span_exported(exported_data, "application", "start")
        self._assert_span_exported(exported_data, "application", "stop")

        # Should have exported training_loop spans (both start and stop since on_app_end succeeded)
        self._assert_span_exported(exported_data, "training_loop", "start")
        self._assert_span_exported(exported_data, "training_loop", "stop")

        # Note: training_single_iteration spans are not exported due to blacklist, even when incomplete
        training_iteration_records = [record for record in exported_data if record.get("name") == "training_single_iteration"]
        assert len(training_iteration_records) == 0, "Training iteration spans should not be exported due to blacklist"

        # Look for training metrics update events (which contain timer data)
        training_metrics_events = [
            record for record in exported_data if record.get("type") == "event" and record.get("event", {}).get("name") == "training_metrics_update"
        ]
        # NOTE: In OneLogger v1, we also don't export timer statistics for iteration 6.
        # Data is only flushed when log_interval is reached or starting save checkpoint.
        assert len(training_metrics_events) == 1, f"Expected 1 training metrics event (at iteration 5), " f"got {len(training_metrics_events)}"

        # Validate the training metrics event (should be at iteration 5)
        metrics_event = training_metrics_events[0]
        event_data = metrics_event.get("event", {})
        attributes = event_data.get("attributes", {})

        # Validate timer-related attributes
        assert "num_iterations" in attributes, "Training metrics should include num_iterations"
        assert "avg_iteration_time_sec" in attributes, "Training metrics should include avg_iteration_time_sec"
        assert "min_iteration_time_sec" in attributes, "Training metrics should include min_iteration_time_sec"
        assert "max_iteration_time_sec" in attributes, "Training metrics should include max_iteration_time_sec"
        assert "total_iteration_time_sec" in attributes, "Training metrics should include total_iteration_time_sec"

        # Validate timer data values
        num_iterations = attributes.get("num_iterations")
        assert num_iterations == 5, f"Metrics event should have 5 iterations, got {num_iterations}"

        avg_time = attributes.get("avg_iteration_time_sec")
        min_time = attributes.get("min_iteration_time_sec")
        max_time = attributes.get("max_iteration_time_sec")
        total_time = attributes.get("total_iteration_time_sec")

        assert avg_time > 0, f"Average iteration time should be positive, got {avg_time}"
        assert min_time > 0, f"Minimum iteration time should be positive, got {min_time}"
        assert max_time > 0, f"Maximum iteration time should be positive, got {max_time}"
        assert total_time > 0, f"Total iteration time should be positive, got {total_time}"
        assert min_time <= avg_time <= max_time, f"Timer statistics should be consistent: min={min_time}, avg={avg_time}, max={max_time}"
        assert (
            abs(total_time - (avg_time * num_iterations)) < 0.001
        ), f"Total time should equal avg * num_iterations: {total_time} vs {avg_time * num_iterations}"

        # Verify that the multi-window timer internal state is correct
        training_iteration_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.TRAINING_SINGLE_ITERATION]
        assert training_iteration_timer.total_window_count == 7, "Timer should have tracked 7 training iterations (6 completed + 1 started but not finished)"
        assert training_iteration_timer.total_time_sec > 0, "Timer should have accumulated total time"
        assert training_iteration_timer.min_window_duration_sec > 0, "Timer should have minimum duration"
        assert training_iteration_timer.max_window_duration_sec > 0, "Timer should have maximum duration"
        assert training_iteration_timer.avg_window_duration_sec > 0, "Timer should have average duration"

        # Verify that the timer statistics in the exported event match the internal timer state
        # Note: The exported event only contains data for the first 5 iterations (log_every_n_train_iterations=5)
        # The internal timer has data for 6 completed iterations, so we need to calculate what the first 5 iterations should be
        # Since we're using varying iteration times (0.1 + i * 0.01), the first 5 iterations should have:
        # iteration 0: 0.1, iteration 1: 0.11, iteration 2: 0.12, iteration 3: 0.13, iteration 4: 0.14
        expected_first_5_total_time = 0.1 + 0.11 + 0.12 + 0.13 + 0.14  # 0.6 seconds
        expected_first_5_avg_time = expected_first_5_total_time / 5  # 0.12 seconds
        expected_first_5_min_time = 0.1  # iteration 0
        expected_first_5_max_time = 0.14  # iteration 4

        # Allow for small floating point differences
        assert (
            abs(total_time - expected_first_5_total_time) < 0.001
        ), f"Exported total time should match expected first 5 iterations: {total_time} vs {expected_first_5_total_time}"
        assert (
            abs(avg_time - expected_first_5_avg_time) < 0.001
        ), f"Exported avg time should match expected first 5 iterations: {avg_time} vs {expected_first_5_avg_time}"
        assert (
            abs(min_time - expected_first_5_min_time) < 0.001
        ), f"Exported min time should match expected first 5 iterations: {min_time} vs {expected_first_5_min_time}"
        assert (
            abs(max_time - expected_first_5_max_time) < 0.001
        ), f"Exported max time should match expected first 5 iterations: {max_time} vs {expected_first_5_max_time}"

        # Verify that the internal timer state reflects all 7 iterations (including the incomplete one that was stopped)
        # When the training loop span was stopped, it automatically stopped the timer for the incomplete iteration
        # The incomplete iteration includes the additional 0.1 seconds we advanced before calling on_app_end()
        # So now we have 7 completed iterations (6 normal + 1 that was force-stopped with extra time)
        expected_all_7_total_time = 0.1 + 0.11 + 0.12 + 0.13 + 0.14 + 0.15 + (0.16 + 0.1)  # 1.01 seconds (including the extra 0.1)
        expected_all_7_avg_time = expected_all_7_total_time / 7  # ~0.144 seconds
        expected_all_7_min_time = 0.1  # iteration 0
        expected_all_7_max_time = 0.26  # iteration 6 (the incomplete one that was stopped, including extra time)

        assert (
            abs(training_iteration_timer.total_time_sec - expected_all_7_total_time) < 0.001
        ), f"Internal timer total time should match all 7 iterations: {training_iteration_timer.total_time_sec} vs {expected_all_7_total_time}"
        assert (
            abs(training_iteration_timer.avg_window_duration_sec - expected_all_7_avg_time) < 0.001
        ), f"Internal timer avg time should match all 7 iterations: {training_iteration_timer.avg_window_duration_sec} vs {expected_all_7_avg_time}"
        assert (
            abs(training_iteration_timer.min_window_duration_sec - expected_all_7_min_time) < 0.001
        ), f"Internal timer min time should match all 7 iterations: {training_iteration_timer.min_window_duration_sec} vs {expected_all_7_min_time}"
        assert (
            abs(training_iteration_timer.max_window_duration_sec - expected_all_7_max_time) < 0.001
        ), f"Internal timer max time should match all 7 iterations: {training_iteration_timer.max_window_duration_sec} vs {expected_all_7_max_time}"

        # Verify that the timer is no longer active since the incomplete iteration was stopped
        assert not training_iteration_timer.is_active, "Timer should not be active since the incomplete iteration was stopped"

        # Should have exported telemetry data error
        error_record = self._assert_telemetry_data_error_exported(
            exported_data,
            "Application crashed before on_training_single_iteration_end at iteration 7",
        )
        error_event = error_record.get("error", {})
        error_attributes = error_event.get("attributes", {})
        assert error_attributes.get("crash_reason") == "simulated_crash_with_app_end_handler"
        assert error_attributes.get("active_spans_count") == 3  # app, training_loop, training_single_iteration
        assert error_attributes.get("crash_iteration") == 7
        assert error_attributes.get("crash_mode") == "timer_data_export_test_with_app_end"

        # Verify that the recorder is closed after on_app_end
        assert training_recorder._closed, "Recorder should be closed after on_app_end"

    def test_timer_data_export_with_missing_first_iteration_end_and_complete_second_iteration(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when the first on_training_single_iteration_end is missing but the second iteration is complete."""
        # Create a custom config with smaller log_every_n_train_iterations for testing
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        _ = TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Start iteration 1 but forget to end it
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1)  # iteration 1 time

        # User forgot to call on_training_single_iteration_end() for iteration 1
        # This should cause an error when trying to start iteration 2
        from nv_one_logger.core.exceptions import OneLoggerError

        try:
            on_training_single_iteration_start()
            # If we get here, the system didn't prevent overlapping iterations (unexpected)
            raise AssertionError("Expected OneLoggerError when trying to start overlapping training iteration")
        except OneLoggerError as e:
            # Expected error: "Cannot start timer since it is already active"
            assert "Cannot start timer since it is already active" in str(e), f"Expected timer already active error, got: {e}"

    def test_timer_data_export_with_missed_checkpoint_load_end(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when the user misses calling on_load_checkpoint_end."""
        # Create a custom config with smaller log_every_n_train_iterations for testing
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,  # Smaller value to trigger more frequent exports
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        training_recorder = TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)

        # Start application
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        # Start checkpoint loading but forget to end it
        on_load_checkpoint_start()
        advance_time(mock_time, mock_perf_counter, 0.5)  # checkpoint loading time

        # User forgot to call on_load_checkpoint_end() and directly starts training loop
        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Do a few training iterations
        for i in range(5):  # iterations 0-4
            on_training_single_iteration_start()
            advance_time(mock_time, mock_perf_counter, 0.1 + i * 0.01)  # Varying iteration times
            on_training_single_iteration_end()
            advance_time(mock_time, mock_perf_counter, 0.02)

        # End training loop and application
        on_train_end()
        advance_time(mock_time, mock_perf_counter, 0.1)
        on_app_end()

        # Verify exported data
        exported_data = self._extract_exported_data()

        # Should have exported application spans (both start and stop)
        self._assert_span_exported(exported_data, "application", "start")
        self._assert_span_exported(exported_data, "application", "stop")

        # Should have exported training_loop spans (both start and stop)
        self._assert_span_exported(exported_data, "training_loop", "start")
        self._assert_span_exported(exported_data, "training_loop", "stop")

        # Should have exported checkpoint_load span (start only, since on_load_checkpoint_end was not called)
        self._assert_span_exported(exported_data, "checkpoint_load", "start")
        # checkpoint_load stop is exported because on_app_end automatically stopped all spans
        checkpoint_load_stop_record = self._assert_span_exported(exported_data, "checkpoint_load", "stop")

        # Verify that the checkpoint load stop time is close to the end of the application
        # The checkpoint load should have been stopped during on_app_end() cleanup
        app_stop_record = self._assert_span_exported(exported_data, "application", "stop")

        # Extract timestamps from the records
        # Based on the actual structure, timestamps are in stop_event.attributes.timestamp_msec (in milliseconds)
        checkpoint_load_stop_time = checkpoint_load_stop_record["stop_event"]["attributes"]["timestamp_msec"]
        app_stop_time = app_stop_record["stop_event"]["attributes"]["timestamp_msec"]

        assert checkpoint_load_stop_time is not None, "Checkpoint load stop time should be present"
        assert app_stop_time is not None, "Application stop time should be present"

        # The checkpoint load stop time should be very close to the app stop time
        # since it was stopped during the on_app_end() cleanup process
        # Convert from milliseconds to seconds for comparison
        # TODO: need to add a fix in nv-one-logger to control which span can be overlapped with other spans.
        time_diff_sec = abs(checkpoint_load_stop_time - app_stop_time) / 1000.0
        assert time_diff_sec < 0.1, f"Checkpoint load stop time should be close to app stop time, difference: {time_diff_sec} seconds"

        # Verify that the checkpoint load timer is no longer active since on_app_end automatically stopped all spans
        checkpoint_load_timer = training_recorder._training_state.multi_iteration_timers[StandardTrainingJobSpanName.CHECKPOINT_LOAD]
        assert not checkpoint_load_timer.is_active, "Checkpoint load timer should not be active since on_app_end automatically stopped all spans"

        # Verify that the recorder is closed after on_app_end
        assert training_recorder._closed, "Recorder should be closed after on_app_end"

    def test_timer_data_export_with_new_iteration_start_before_old_iteration_end(
        self,
        training_recorder: TrainingRecorder,
        mock_perf_counter: Mock,
        mock_time: Mock,
    ) -> None:
        """Test that timer data is exported correctly when the user misses calling on_training_single_iteration_end for iteration 1."""
        # Create a custom config with smaller log_every_n_train_iterations for testing
        from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig

        custom_config = TrainingTelemetryConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            enable_one_logger=True,
            enable_for_current_rank=True,
            is_save_checkpoint_enabled_or_fn=True,
            is_log_throughput_enabled_or_fn=True,
            training_loop_config=TrainingLoopConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=10,
                flops_per_sample_or_fn=100,
                global_batch_size_or_fn=32,
                log_every_n_train_iterations=5,  # Smaller value to trigger more frequent exports
                train_iterations_target_or_fn=25,
                train_samples_target_or_fn=800,
            ),
        )

        # Reconfigure the provider with the custom config
        from .conftest import configure_provider_for_test

        configure_provider_for_test(custom_config, self.exporter)

        # Get a fresh training recorder with the new config
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

        TrainingTelemetryProvider.instance().recorder

        # Initialize time to a valid value
        advance_time(mock_time, mock_perf_counter, 1000.0)  # Start at 1000 seconds

        # Start application and training loop
        on_app_start()
        advance_time(mock_time, mock_perf_counter, 1.0)

        on_train_start(
            train_iterations_start=0,
            train_iterations_target_or_fn=25,
            train_samples_target_or_fn=800,
        )
        advance_time(mock_time, mock_perf_counter, 0.5)

        # Start iteration 1 but forget to end it
        on_training_single_iteration_start()
        advance_time(mock_time, mock_perf_counter, 0.1)  # iteration 1 time

        # User forgot to call on_training_single_iteration_end() for iteration 1
        # and tries to directly start iteration 2 - this should fail
        from nv_one_logger.core.exceptions import OneLoggerError

        try:
            on_training_single_iteration_start()
            # If we get here, the system didn't prevent overlapping iterations (unexpected)
            raise AssertionError("Expected OneLoggerError when trying to start overlapping training iteration")
        except OneLoggerError as e:
            # Expected error: "Cannot start timer since it is already active"
            assert "Cannot start timer since it is already active" in str(e), f"Expected timer already active error, got: {e}"
