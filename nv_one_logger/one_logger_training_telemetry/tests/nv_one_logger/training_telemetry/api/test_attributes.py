# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the attributes module."""

import pytest

from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.training_telemetry.api.attributes import (
    CheckpointSaveSpanAttributes,
    OneLoggerInitializationAttributes,
    SaveCheckpointSuccessEventAttributes,
    TestingMetricsUpdateAttributes,
    TrainingLoopAttributes,
    TrainingMetricsUpdateAttributes,
    TrainingTelemetryAttributes,
    ValidationMetricsUpdateAttributes,
)
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy


class TestTrainingLoopAttributes:
    """Tests for TrainingLoopAttributes class."""

    def test_create_with_required_parameters(self) -> None:
        """Test creating TrainingLoopAttributes with only required parameters."""
        attrs = TrainingLoopAttributes.create(
            train_iterations_start=100,
            train_samples_start=1000,
        )
        assert attrs.train_iterations_start == 100
        assert attrs.train_samples_start == 1000
        assert attrs.train_iterations_target is None
        assert attrs.train_samples_target is None
        assert attrs.train_tokens_target is None

    def test_create_with_all_parameters(self) -> None:
        """Test creating TrainingLoopAttributes with all parameters."""
        attrs = TrainingLoopAttributes.create(
            train_iterations_start=100,
            train_samples_start=1000,
            train_iterations_target=1000,
            train_samples_target=10000,
            train_tokens_target=50000,
            completed_floating_point_operations_overall=1000000,
        )
        assert attrs.train_iterations_start == 100
        assert attrs.train_samples_start == 1000
        assert attrs.train_iterations_target == 1000
        assert attrs.train_samples_target == 10000
        assert attrs.train_tokens_target == 50000
        assert attrs.completed_floating_point_operations_overall == 1000000

    def test_pass_none_for_required_parameter(self) -> None:
        """Test that passing None for a required parameter raises an error."""
        with pytest.raises(OneLoggerError, match="train_iterations_start is required"):
            TrainingLoopAttributes.create(
                train_iterations_start=None,  # type: ignore
                train_samples_start=1000,
            )


class TestCheckpointSaveSpanAttributes:
    """Tests for CheckpointSaveSpanAttributes class."""

    def test_create(self) -> None:
        """Test creating CheckpointSaveSpanAttributes with all required parameters."""
        attrs = CheckpointSaveSpanAttributes.create(
            checkpoint_strategy=CheckPointStrategy.SYNC,
            current_iteration=100,
            save_checkpoint_attempt_count=5,
        )
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC
        assert attrs.current_iteration == 100
        assert attrs.save_checkpoint_attempt_count == 5

    def test_pass_none_for_checkpoint_strategy(self) -> None:
        """Test that passing None for checkpoint_strategy raises an error."""
        with pytest.raises(OneLoggerError, match="checkpoint_strategy is required"):
            CheckpointSaveSpanAttributes.create(
                checkpoint_strategy=None,  # type: ignore
                current_iteration=100,
                save_checkpoint_attempt_count=5,
            )

    def test_pass_none_for_current_iteration(self) -> None:
        """Test that passing None for current_iteration raises an error."""
        with pytest.raises(OneLoggerError, match="current_iteration is required"):
            CheckpointSaveSpanAttributes.create(
                checkpoint_strategy=CheckPointStrategy.SYNC,
                current_iteration=None,  # type: ignore
                save_checkpoint_attempt_count=5,
            )

    def test_pass_none_for_save_checkpoint_attempt_count(self) -> None:
        """Test that passing None for save_checkpoint_attempt_count raises an error."""
        with pytest.raises(OneLoggerError, match="save_checkpoint_attempt_count is required"):
            CheckpointSaveSpanAttributes.create(
                checkpoint_strategy=CheckPointStrategy.SYNC,
                current_iteration=100,
                save_checkpoint_attempt_count=None,  # type: ignore
            )

    def test_property_accessors(self) -> None:
        """Test that property accessors return the correct values."""
        attrs = CheckpointSaveSpanAttributes.create(
            checkpoint_strategy=CheckPointStrategy.ASYNC,
            current_iteration=200,
            save_checkpoint_attempt_count=10,
        )
        assert attrs.checkpoint_strategy == CheckPointStrategy.ASYNC
        assert attrs.current_iteration == 200
        assert attrs.save_checkpoint_attempt_count == 10

    def test_property_accessors_missing_values(self) -> None:
        """Test that property accessors raise errors when values are missing."""
        attrs = CheckpointSaveSpanAttributes()
        with pytest.raises(OneLoggerError, match="checkpoint_strategy is required"):
            _ = attrs.checkpoint_strategy
        with pytest.raises(OneLoggerError, match="current_iteration is required"):
            _ = attrs.current_iteration
        with pytest.raises(OneLoggerError, match="save_checkpoint_attempt_count is required"):
            _ = attrs.save_checkpoint_attempt_count


class TestOneLoggerInitializationAttributes:
    """Tests for OneLoggerInitializationAttributes class."""

    def test_create_with_required_parameters(self) -> None:
        """Test creating OneLoggerInitializationAttributes with only required parameters."""
        attrs = OneLoggerInitializationAttributes.create(
            world_size=4,
            one_logger_training_telemetry_version="1.0.0",
            enable_for_current_rank=True,
            session_tag="test_session",
            is_baseline_run=False,
            summary_data_schema_version="1.0",
            node_name="test_node",
            rank=0,
        )
        assert attrs.one_logger_training_telemetry_version == "1.0.0"
        assert attrs.enable_for_current_rank is True
        assert attrs.session_tag == "test_session"
        assert attrs.is_baseline_run is False
        assert attrs.summary_data_schema_version == "1.0"
        assert attrs.node_name == "test_node"
        assert attrs.rank == 0
        assert attrs.custom_metadata is None

    def test_create_with_optional_parameters(self) -> None:
        """Test creating OneLoggerInitializationAttributes with optional parameters."""
        attrs = OneLoggerInitializationAttributes.create(
            world_size=4,
            one_logger_training_telemetry_version="1.0.0",
            enable_for_current_rank=True,
            session_tag="test_session",
            is_baseline_run=False,
            summary_data_schema_version="1.0",
            node_name="test_node",
            rank=0,
            custom_metadata={"key1": "value1", "key2": "value2"},
        )
        assert attrs.custom_metadata == ["key1:value1", "key2:value2"]

    def test_pass_none_for_required_parameters(self) -> None:
        """Test that passing None for any required parameter raises an error."""
        required_params = {
            "one_logger_training_telemetry_version": "one_logger_training_telemetry_version is required",
            "enable_for_current_rank": "enable_for_current_rank is required",
            "session_tag": "session_tag is required",
            "is_baseline_run": "is_baseline_run is required",
            "summary_data_schema_version": "summary_data_schema_version is required",
            "node_name": "node_name is required",
            "rank": "rank is required",
        }

        base_params = {
            "world_size": 4,
            "one_logger_training_telemetry_version": "1.0.0",
            "enable_for_current_rank": True,
            "session_tag": "test_session",
            "is_baseline_run": False,
            "summary_data_schema_version": "1.0",
            "node_name": "test_node",
            "rank": 0,
        }

        for param, error_msg in required_params.items():
            params = base_params.copy()
            params[param] = None  # type: ignore[assignment]
            with pytest.raises(OneLoggerError, match=error_msg):
                OneLoggerInitializationAttributes.create(**params)


class TestTrainingMetricsUpdateAttributes:
    """Tests for TrainingMetricsUpdateAttributes class."""

    def test_create_with_required_parameters(self) -> None:
        """Test creating TrainingMetricsUpdateAttributes with only required parameters."""
        attrs = TrainingMetricsUpdateAttributes.create(
            train_iterations_start=100,
            current_iteration=200,
            num_iterations=100,
            train_samples_start=1000,
            num_train_samples=2000,
            interval=100,
            avg_iteration_time_sec=0.1,
            min_iteration_time_sec=0.05,
            max_iteration_time_sec=0.15,
            total_iteration_time_sec=10.0,
        )
        assert attrs.train_iterations_start == 100
        assert attrs.current_iteration == 200
        assert attrs.num_iterations == 100
        assert attrs.train_samples_start == 1000
        assert attrs.num_train_samples == 2000
        assert attrs.interval == 100
        assert attrs.avg_iteration_time_sec == 0.1
        assert attrs.min_iteration_time_sec == 0.05
        assert attrs.max_iteration_time_sec == 0.15
        assert attrs.total_iteration_time_sec == 10.0
        assert attrs.avg_forward_time_sec is None
        assert attrs.avg_backward_time_sec is None
        assert attrs.avg_dataloader_time_sec is None
        assert attrs.avg_tflops is None
        assert attrs.train_tokens is None
        assert attrs.avg_tokens_per_second is None
        assert attrs.latest_loss is None
        assert attrs.avg_batch_size is None
        assert attrs.completed_floating_point_operations_overall is None
        assert attrs.total_flops is None
        assert attrs.train_throughput_per_gpu is None
        assert attrs.train_throughput_per_gpu_max is None
        assert attrs.train_throughput_per_gpu_min is None

    def test_create_with_all_parameters(self) -> None:
        """Test creating TrainingMetricsUpdateAttributes with all parameters."""
        attrs = TrainingMetricsUpdateAttributes.create(
            train_iterations_start=100,
            current_iteration=200,
            num_iterations=100,
            train_samples_start=1000,
            num_train_samples=2000,
            interval=100,
            avg_iteration_time_sec=0.1,
            min_iteration_time_sec=0.05,
            max_iteration_time_sec=0.15,
            total_iteration_time_sec=10.0,
            avg_forward_time_sec=0.05,
            avg_backward_time_sec=0.03,
            avg_dataloader_time_sec=0.02,
            avg_tflops=100.0,
            train_tokens=50000,
            avg_tokens_per_second=5000.0,
            latest_loss=0.5,
            avg_batch_size=32,
            completed_floating_point_operations_overall=1000000,
            total_flops=500000,
            train_throughput_per_gpu=90.0,
            train_throughput_per_gpu_max=100.0,
            train_throughput_per_gpu_min=80.0,
        )
        assert attrs.avg_iteration_time_sec == 0.1
        assert attrs.min_iteration_time_sec == 0.05
        assert attrs.max_iteration_time_sec == 0.15
        assert attrs.total_iteration_time_sec == 10.0
        assert attrs.avg_forward_time_sec == 0.05
        assert attrs.avg_backward_time_sec == 0.03
        assert attrs.avg_dataloader_time_sec == 0.02
        assert attrs.avg_tflops == 100.0
        assert attrs.train_tokens == 50000
        assert attrs.avg_tokens_per_second == 5000.0
        assert attrs.latest_loss == 0.5
        assert attrs.avg_batch_size == 32
        assert attrs.completed_floating_point_operations_overall == 1000000
        assert attrs.total_flops == 500000
        assert attrs.train_throughput_per_gpu == 90.0
        assert attrs.train_throughput_per_gpu_max == 100.0
        assert attrs.train_throughput_per_gpu_min == 80.0

    def test_pass_none_for_required_parameters(self) -> None:
        """Test that passing None for any required parameter raises an error."""
        required_params = {
            "train_iterations_start": "train_iterations_start is required",
            "current_iteration": "current_iteration is required",
            "num_iterations": "num_iterations is required",
            "train_samples_start": "train_samples_start is required",
            "num_train_samples": "num_train_samples is required",
            "interval": "interval is required",
            "avg_iteration_time_sec": "avg_iteration_time_sec is required",
            "min_iteration_time_sec": "min_iteration_time_sec is required",
            "max_iteration_time_sec": "max_iteration_time_sec is required",
            "total_iteration_time_sec": "total_iteration_time_sec is required",
        }

        base_params = {
            "train_iterations_start": 100,
            "current_iteration": 200,
            "num_iterations": 100,
            "train_samples_start": 1000,
            "num_train_samples": 2000,
            "interval": 100,
            "avg_iteration_time_sec": 0.1,
            "min_iteration_time_sec": 0.05,
            "max_iteration_time_sec": 0.15,
            "total_iteration_time_sec": 10.0,
        }

        for param, error_msg in required_params.items():
            params = base_params.copy()
            params[param] = None  # type: ignore[assignment]
            with pytest.raises(OneLoggerError, match=error_msg):
                TrainingMetricsUpdateAttributes.create(**params)  # type: ignore[arg-type]


class TestValidationMetricsUpdateAttributes:
    """Tests for ValidationMetricsUpdateAttributes class."""

    def test_create_with_required_parameters(self) -> None:
        """Test creating ValidationMetricsUpdateAttributes."""
        attrs = ValidationMetricsUpdateAttributes.create(
            current_iteration=100,
            interval=50,
            avg_iteration_time_sec=0.1,
            min_iteration_time_sec=0.05,
            max_iteration_time_sec=0.15,
            total_iteration_time_sec=10.0,
        )
        assert attrs.current_iteration == 100
        assert attrs.interval == 50
        assert attrs.avg_iteration_time_sec == 0.1
        assert attrs.min_iteration_time_sec == 0.05
        assert attrs.max_iteration_time_sec == 0.15
        assert attrs.total_iteration_time_sec == 10.0

    def test_create_with_all_parameters(self) -> None:
        """Test creating ValidationMetricsUpdateAttributes with all parameters."""
        attrs = ValidationMetricsUpdateAttributes.create(
            current_iteration=100,
            interval=50,
            avg_iteration_time_sec=0.1,
            min_iteration_time_sec=0.05,
            max_iteration_time_sec=0.15,
            total_iteration_time_sec=10.0,
        )
        assert attrs.current_iteration == 100
        assert attrs.interval == 50
        assert attrs.avg_iteration_time_sec == 0.1
        assert attrs.min_iteration_time_sec == 0.05
        assert attrs.max_iteration_time_sec == 0.15
        assert attrs.total_iteration_time_sec == 10.0

    def test_pass_none_for_required_parameters(self) -> None:
        """Test that passing None for any required parameter raises an error."""
        required_params = {
            "current_iteration": "current_iteration is required",
            "interval": "interval is required",
        }

        base_params = {
            "current_iteration": 100,
            "interval": 50,
        }

        for param, error_msg in required_params.items():
            params = base_params.copy()
            params[param] = None  # type: ignore[assignment]
            with pytest.raises(OneLoggerError, match=error_msg):
                ValidationMetricsUpdateAttributes.create(**params)


class TestTestingMetricsUpdateAttributes:
    """Tests for TestingMetricsUpdateAttributes class."""

    def test_create(self) -> None:
        """Test creating TestingMetricsUpdateAttributes with all parameters."""
        attrs = TestingMetricsUpdateAttributes.create(
            current_iteration=100,
            interval=50,
        )
        assert attrs.current_iteration == 100
        assert attrs.interval == 50

    def test_pass_none_for_required_parameters(self) -> None:
        """Test that passing None for any required parameter raises an error."""
        required_params = {
            "current_iteration": "current_iteration is required",
            "interval": "interval is required",
        }

        base_params = {
            "current_iteration": 100,
            "interval": 50,
        }

        for param, error_msg in required_params.items():
            params = base_params.copy()
            params[param] = None  # type: ignore[assignment]
            with pytest.raises(OneLoggerError, match=error_msg):
                TestingMetricsUpdateAttributes.create(**params)


class TestSaveCheckpointSuccessEventAttributes:
    """Tests for SaveCheckpointSuccessEventAttributes class."""

    def test_create_with_required_parameters(self) -> None:
        """Test creating SaveCheckpointSuccessEventAttributes with only required parameters."""
        attrs = SaveCheckpointSuccessEventAttributes.create(
            checkpoint_strategy=CheckPointStrategy.SYNC,
            current_iteration=100,
            first_successful_save_checkpoint_timestamp_sec=1100.0,
            latest_successful_save_checkpoint_timestamp_sec=1100.0,
            save_checkpoint_success_count=5,
            productive_train_iterations=50,
            productive_train_samples=500,
            productive_train_iterations_sec=30.0,
            productive_validation_iterations_sec=10.0,
        )
        assert isinstance(attrs.checkpoint_strategy, CheckPointStrategy)
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC
        assert attrs.current_iteration == 100
        assert attrs.first_successful_save_checkpoint_timestamp_sec == 1100.0
        assert attrs.latest_successful_save_checkpoint_timestamp_sec == 1100.0
        assert attrs.save_checkpoint_success_count == 5
        assert attrs.productive_train_iterations == 50
        assert attrs.productive_train_samples == 500
        assert attrs.productive_train_iterations_sec == 30.0
        assert attrs.productive_validation_iterations_sec == 10.0
        assert attrs.checkpoint_size is None
        assert attrs.checkpoint_directory is None
        assert attrs.training_start_timestamp_sec is None

    def test_create_with_all_parameters(self) -> None:
        """Test creating SaveCheckpointSuccessEventAttributes with all parameters."""
        attrs = SaveCheckpointSuccessEventAttributes.create(
            checkpoint_strategy=CheckPointStrategy.ASYNC,
            current_iteration=200,
            first_successful_save_checkpoint_timestamp_sec=2100.0,
            latest_successful_save_checkpoint_timestamp_sec=2100.0,
            save_checkpoint_success_count=5,
            productive_train_iterations=100,
            productive_train_samples=1000,
            productive_train_tflops=50.0,
            productive_train_iterations_sec=60.0,
            productive_validation_iterations_sec=20.0,
            checkpoint_size=1000000,
            checkpoint_directory="/path/to/checkpoint",
            training_start_timestamp_sec=2000.0,
        )
        assert attrs.checkpoint_strategy == CheckPointStrategy.ASYNC
        assert attrs.current_iteration == 200
        assert attrs.first_successful_save_checkpoint_timestamp_sec == 2100.0
        assert attrs.latest_successful_save_checkpoint_timestamp_sec == 2100.0
        assert attrs.save_checkpoint_success_count == 5
        assert attrs.productive_train_iterations == 100
        assert attrs.productive_train_samples == 1000
        assert attrs.productive_train_tflops == 50.0
        assert attrs.productive_train_iterations_sec == 60.0
        assert attrs.productive_validation_iterations_sec == 20.0
        assert attrs.checkpoint_size == 1000000
        assert attrs.checkpoint_directory == "/path/to/checkpoint"
        assert attrs.training_start_timestamp_sec == 2000.0

    def test_pass_none_for_required_parameters(self) -> None:
        """Test that passing None for any required parameter raises an error."""
        required_params = {
            "checkpoint_strategy": "checkpoint_strategy is required",
            "current_iteration": "current_iteration is required",
            "first_successful_save_checkpoint_timestamp_sec": "first_successful_save_checkpoint_timestamp_sec is required",
            "latest_successful_save_checkpoint_timestamp_sec": "latest_successful_save_checkpoint_timestamp_sec is required",
            "save_checkpoint_success_count": "save_checkpoint_success_count is required",
            "productive_train_iterations": "productive_train_iterations is required",
            "productive_train_samples": "productive_train_samples is required",
            "productive_train_iterations_sec": "productive_train_iterations_sec is required",
            "productive_validation_iterations_sec": "productive_validation_iterations_sec is required",
        }

        base_params = {
            "checkpoint_strategy": CheckPointStrategy.SYNC,
            "current_iteration": 100,
            "first_successful_save_checkpoint_timestamp_sec": 1100.0,
            "latest_successful_save_checkpoint_timestamp_sec": 1100.0,
            "save_checkpoint_success_count": 2,
            "productive_train_iterations": 50,
            "productive_train_samples": 500,
            "productive_train_iterations_sec": 30.0,
            "productive_validation_iterations_sec": 10.0,
            "productive_train_tflops": 25.0,
            "checkpoint_size": 400,
            "checkpoint_directory": "/path/to/checkpoint",
            "training_start_timestamp_sec": 2000.0,
        }

        for param, error_msg in required_params.items():
            params = base_params.copy()
            params[param] = None  # type: ignore[assignment]
            with pytest.raises(OneLoggerError, match=error_msg):
                SaveCheckpointSuccessEventAttributes.create(**params)


class TestTrainingTelemetryAttributes:
    """Test cases for TrainingTelemetryAttributes class."""

    def test_create_with_all_required_fields(self):
        """Test creating TrainingTelemetryAttributes with all required fields."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
        )

        assert attrs.perf_tag == "test_perf"
        assert attrs.global_batch_size == 64
        assert attrs.log_every_n_train_iterations == 10

    def test_create_with_all_fields(self):
        """Test creating TrainingTelemetryAttributes with all fields."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            micro_batch_size=32,
            seq_length=512,
            flops_per_sample=1000,
            train_iterations_target=1000,
            train_samples_target=100000,
            checkpoint_strategy=CheckPointStrategy.SYNC,
            is_train_iterations_enabled=True,
            is_validation_iterations_enabled=True,
            is_test_iterations_enabled=True,
            is_save_checkpoint_enabled=True,
            is_log_throughput_enabled=True,
        )

        assert attrs.perf_tag == "test_perf"
        assert attrs.global_batch_size == 64
        assert attrs.log_every_n_train_iterations == 10
        assert attrs.micro_batch_size == 32
        assert attrs.seq_length == 512
        assert attrs.flops_per_sample == 1000
        assert attrs.train_iterations_target == 1000
        assert attrs.train_samples_target == 100000
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC
        assert attrs.is_train_iterations_enabled is True
        assert attrs.is_validation_iterations_enabled is True
        assert attrs.is_test_iterations_enabled is True
        assert attrs.is_save_checkpoint_enabled is True
        assert attrs.is_log_throughput_enabled is True

    def test_create_with_custom_metadata(self):
        """Test creating TrainingTelemetryAttributes with telemetry metadata."""
        custom_metadata = {"telemetry_key1": "telemetry_value1"}

        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            custom_metadata=custom_metadata,
        )

        assert attrs.custom_metadata == ["telemetry_key1:telemetry_value1"]

    def test_create_with_perf_tag_list(self):
        """Test creating TrainingTelemetryAttributes with perf_tag as a list."""
        perf_tags = ["tag1", "tag2", "tag3"]
        attrs = TrainingTelemetryAttributes.create(
            perf_tag=perf_tags,
            global_batch_size=64,
            log_every_n_train_iterations=10,
        )

        assert attrs.perf_tag == perf_tags

    def test_create_with_optional_fields_none(self):
        """Test creating TrainingTelemetryAttributes with optional fields as None."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            # All optional fields are None by default
        )

        # Required fields should be present
        assert attrs.perf_tag == "test_perf"
        assert attrs.global_batch_size == 64
        assert attrs.log_every_n_train_iterations == 10

        # Optional fields should be None
        assert attrs.micro_batch_size is None
        assert attrs.seq_length is None
        assert attrs.flops_per_sample is None
        assert attrs.train_iterations_target is None
        assert attrs.train_samples_target is None
        assert attrs.checkpoint_strategy is None
        assert attrs.is_train_iterations_enabled is None
        assert attrs.is_validation_iterations_enabled is None
        assert attrs.is_test_iterations_enabled is None
        assert attrs.is_save_checkpoint_enabled is None
        assert attrs.is_log_throughput_enabled is None
        assert attrs.custom_metadata is None

    def test_create_missing_required_fields(self):
        """Test that creating with missing required fields raises errors."""
        # Test missing perf_tag - this will fail at the method signature level
        with pytest.raises(TypeError):
            TrainingTelemetryAttributes.create(
                global_batch_size=64,
                log_every_n_train_iterations=10,
            )

        # Test missing global_batch_size - this will fail at the method signature level
        with pytest.raises(TypeError):
            TrainingTelemetryAttributes.create(
                perf_tag="test_perf",
                log_every_n_train_iterations=10,
            )

        # Test missing log_every_n_train_iterations - this will fail at the method signature level
        with pytest.raises(TypeError):
            TrainingTelemetryAttributes.create(
                perf_tag="test_perf",
                global_batch_size=64,
            )

    def test_create_with_none_required_fields(self):
        """Test that creating with None required fields raises errors."""
        # Test None perf_tag
        with pytest.raises(OneLoggerError, match="perf_tag is required\\."):
            TrainingTelemetryAttributes.create(
                perf_tag=None,  # type: ignore
                global_batch_size=64,
                log_every_n_train_iterations=10,
            )

        # Test None global_batch_size
        with pytest.raises(OneLoggerError, match="global_batch_size is required\\."):
            TrainingTelemetryAttributes.create(
                perf_tag="test_perf",
                global_batch_size=None,  # type: ignore
                log_every_n_train_iterations=10,
            )

        # Test None log_every_n_train_iterations
        with pytest.raises(OneLoggerError, match="log_every_n_train_iterations is required\\."):
            TrainingTelemetryAttributes.create(
                perf_tag="test_perf",
                global_batch_size=64,
                log_every_n_train_iterations=None,  # type: ignore
            )

    def test_property_access_required_fields(self):
        """Test property access for required fields."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
        )

        # Test property access
        assert attrs.perf_tag == "test_perf"
        assert attrs.global_batch_size == 64
        assert attrs.log_every_n_train_iterations == 10

    def test_property_access_optional_fields(self):
        """Test property access for optional fields."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            micro_batch_size=32,
            seq_length=512,
            flops_per_sample=1000,
            train_iterations_target=1000,
            train_samples_target=100000,
            checkpoint_strategy=CheckPointStrategy.SYNC,
            is_train_iterations_enabled=True,
            is_validation_iterations_enabled=True,
            is_test_iterations_enabled=True,
            is_save_checkpoint_enabled=True,
            is_log_throughput_enabled=True,
        )

        # Test property access for optional fields
        assert attrs.micro_batch_size == 32
        assert attrs.seq_length == 512
        assert attrs.flops_per_sample == 1000
        assert attrs.train_iterations_target == 1000
        assert attrs.train_samples_target == 100000
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC
        assert attrs.is_train_iterations_enabled is True
        assert attrs.is_validation_iterations_enabled is True
        assert attrs.is_test_iterations_enabled is True
        assert attrs.is_save_checkpoint_enabled is True
        assert attrs.is_log_throughput_enabled is True

    def test_property_access_optional_fields_none(self):
        """Test property access for optional fields when they are None."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
        )

        # Test property access for optional fields when None
        assert attrs.micro_batch_size is None
        assert attrs.seq_length is None
        assert attrs.flops_per_sample is None
        assert attrs.train_iterations_target is None
        assert attrs.train_samples_target is None
        assert attrs.checkpoint_strategy is None
        assert attrs.is_train_iterations_enabled is None
        assert attrs.is_validation_iterations_enabled is None
        assert attrs.is_test_iterations_enabled is None
        assert attrs.is_save_checkpoint_enabled is None
        assert attrs.is_log_throughput_enabled is None

    def test_checkpoint_strategy_with_save_disabled(self):
        """Test that checkpoint_strategy is stored even when save_checkpoint_enabled is False."""
        attrs = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            is_save_checkpoint_enabled=False,
            checkpoint_strategy=CheckPointStrategy.SYNC,  # This should be stored regardless
        )

        assert attrs.is_save_checkpoint_enabled is False
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC


class TestTrainingLoopAttributesUpdated:
    """Test cases for the updated TrainingLoopAttributes class."""

    def test_training_loop_attributes_only_contains_loop_specific_fields(self):
        """Test that TrainingLoopAttributes only contains training-loop-specific fields."""
        attrs = TrainingLoopAttributes.create(
            train_iterations_start=100,
            train_samples_start=1000,
            train_tokens_target=50000,
            completed_floating_point_operations_overall=1000000,
            train_iterations_target=1000,
            train_samples_target=10000,
        )

        # Verify only training-loop-specific fields are present
        assert attrs.train_iterations_start == 100
        assert attrs.train_samples_start == 1000
        assert attrs.train_tokens_target == 50000
        assert attrs.completed_floating_point_operations_overall == 1000000
        assert attrs.train_iterations_target == 1000
        assert attrs.train_samples_target == 10000

        # Verify that training configuration fields are NOT present
        # These should now be in TrainingTelemetryAttributes
        assert not hasattr(attrs, "perf_tag")
        assert not hasattr(attrs, "world_size")
        assert not hasattr(attrs, "global_batch_size")
        assert not hasattr(attrs, "log_every_n_train_iterations")
        assert not hasattr(attrs, "micro_batch_size")
        assert not hasattr(attrs, "seq_length")
        assert not hasattr(attrs, "checkpoint_strategy")
        assert not hasattr(attrs, "is_train_iterations_enabled")
        assert not hasattr(attrs, "is_validation_iterations_enabled")
        assert not hasattr(attrs, "is_test_iterations_enabled")
        assert not hasattr(attrs, "is_save_checkpoint_enabled")
        assert not hasattr(attrs, "is_log_throughput_enabled")

    def test_training_loop_attributes_with_optional_fields_none(self):
        """Test TrainingLoopAttributes with optional fields as None."""
        attrs = TrainingLoopAttributes.create(
            train_iterations_start=100,
            train_samples_start=1000,
            # All optional fields are None by default
        )

        # Required fields should be present
        assert attrs.train_iterations_start == 100
        assert attrs.train_samples_start == 1000

        # Optional fields should be None
        assert attrs.train_tokens_target is None
        assert attrs.completed_floating_point_operations_overall is None
        assert attrs.train_iterations_target is None
        assert attrs.train_samples_target is None

    def test_training_loop_attributes_property_access(self):
        """Test property access for TrainingLoopAttributes."""
        attrs = TrainingLoopAttributes.create(
            train_iterations_start=100,
            train_samples_start=1000,
            train_tokens_target=50000,
            completed_floating_point_operations_overall=1000000,
            train_iterations_target=1000,
            train_samples_target=10000,
        )

        # Test property access
        assert attrs.train_iterations_start == 100
        assert attrs.train_samples_start == 1000
        assert attrs.train_tokens_target == 50000
        assert attrs.completed_floating_point_operations_overall == 1000000
        assert attrs.train_iterations_target == 1000
        assert attrs.train_samples_target == 10000


class TestOneLoggerInitializationAttributesUpdated:
    """Test cases for the updated OneLoggerInitializationAttributes class."""

    def test_one_logger_initialization_attributes_only_contains_base_fields(self):
        """Test that OneLoggerInitializationAttributes only contains base config fields."""
        attrs = OneLoggerInitializationAttributes.create(
            world_size=4,
            one_logger_training_telemetry_version="2.3.0",
            enable_for_current_rank=True,
            session_tag="test_session",
            is_baseline_run=False,
            summary_data_schema_version="1.0",
            rank=0,
            node_name="test_node",
            custom_metadata={"key": "value"},
        )

        # Verify only base config fields are present
        assert attrs.one_logger_training_telemetry_version == "2.3.0"
        assert attrs.enable_for_current_rank is True
        assert attrs.session_tag == "test_session"
        assert attrs.is_baseline_run is False
        assert attrs.summary_data_schema_version == "1.0"
        assert attrs.rank == 0
        assert attrs.node_name == "test_node"
        assert attrs.custom_metadata == ["key:value"]

        # Verify that training-specific fields are NOT present
        # These should now be in TrainingTelemetryAttributes
        assert not hasattr(attrs, "is_train_iterations_enabled")
        assert not hasattr(attrs, "is_validation_iterations_enabled")
        assert not hasattr(attrs, "is_test_iterations_enabled")
        assert not hasattr(attrs, "is_save_checkpoint_enabled")
        assert not hasattr(attrs, "is_log_throughput_enabled")
        assert not hasattr(attrs, "checkpoint_strategy")

    def test_one_logger_initialization_attributes_without_custom_metadata(self):
        """Test OneLoggerInitializationAttributes without custom_metadata."""
        attrs = OneLoggerInitializationAttributes.create(
            world_size=4,
            one_logger_training_telemetry_version="2.3.0",
            enable_for_current_rank=True,
            session_tag="test_session",
            is_baseline_run=False,
            summary_data_schema_version="1.0",
            rank=0,
            node_name="test_node",
            # custom_metadata not provided
        )

        # Verify custom_metadata is None
        assert attrs.custom_metadata is None

    def test_one_logger_initialization_attributes_property_access(self):
        """Test property access for OneLoggerInitializationAttributes."""
        attrs = OneLoggerInitializationAttributes.create(
            world_size=4,
            one_logger_training_telemetry_version="2.3.0",
            enable_for_current_rank=True,
            session_tag="test_session",
            is_baseline_run=False,
            summary_data_schema_version="1.0",
            rank=0,
            node_name="test_node",
            custom_metadata={"key1": "value1", "key2": "value2"},
        )

        # Test property access
        assert attrs.one_logger_training_telemetry_version == "2.3.0"
        assert attrs.enable_for_current_rank is True
        assert attrs.session_tag == "test_session"
        assert attrs.is_baseline_run is False
        assert attrs.summary_data_schema_version == "1.0"
        assert attrs.rank == 0
        assert attrs.node_name == "test_node"
        assert attrs.custom_metadata == ["key1:value1", "key2:value2"]
