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
        )
        assert attrs.train_iterations_start == 100
        assert attrs.train_samples_start == 1000
        assert attrs.train_iterations_target == 1000
        assert attrs.train_samples_target == 10000
        assert attrs.train_tokens_target == 50000

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
            one_logger_training_telemetry_version="1.0.0",
            enable_for_current_rank=True,
            perf_tag="test_tag",
            session_tag="test_session",
            app_type="training",
            log_every_n_train_iterations=100,
            world_size=1,
            global_batch_size=32,
            is_baseline_run=False,
            is_train_iterations_enabled=True,
            is_validation_iterations_enabled=True,
            is_test_iterations_enabled=True,
            is_save_checkpoint_enabled=True,
            is_log_throughput_enabled=True,
            summary_data_schema_version="1.0",
            node_name="test_node",
            rank=0,
            checkpoint_strategy=CheckPointStrategy.SYNC,
        )
        assert attrs.one_logger_training_telemetry_version == "1.0.0"
        assert attrs.enable_for_current_rank is True
        assert attrs.perf_tag == "test_tag"
        assert attrs.session_tag == "test_session"
        assert attrs.app_type == "training"
        assert attrs.log_every_n_train_iterations == 100
        assert attrs.world_size == 1
        assert attrs.global_batch_size == 32
        assert attrs.is_baseline_run is False
        assert attrs.is_train_iterations_enabled is True
        assert attrs.is_validation_iterations_enabled is True
        assert attrs.is_test_iterations_enabled is True
        assert attrs.is_save_checkpoint_enabled is True
        assert attrs.is_log_throughput_enabled is True
        assert attrs.summary_data_schema_version == "1.0"
        assert attrs.node_name == "test_node"
        assert attrs.rank == 0
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC
        assert attrs.micro_batch_size is None
        assert attrs.seq_length is None
        assert attrs.custom_metadata is None

    def test_create_with_optional_parameters(self) -> None:
        """Test creating OneLoggerInitializationAttributes with optional parameters."""
        attrs = OneLoggerInitializationAttributes.create(
            one_logger_training_telemetry_version="1.0.0",
            enable_for_current_rank=True,
            perf_tag="test_tag",
            session_tag="test_session",
            app_type="training",
            log_every_n_train_iterations=100,
            world_size=1,
            global_batch_size=32,
            is_baseline_run=False,
            is_train_iterations_enabled=True,
            is_validation_iterations_enabled=True,
            is_test_iterations_enabled=True,
            is_save_checkpoint_enabled=True,
            is_log_throughput_enabled=True,
            summary_data_schema_version="1.0",
            node_name="test_node",
            rank=0,
            checkpoint_strategy=CheckPointStrategy.SYNC,
            micro_batch_size=16,
            seq_length=512,
            custom_metadata={"key1": "value1", "key2": "value2"},
        )
        assert attrs.micro_batch_size == 16
        assert attrs.seq_length == 512
        assert attrs.custom_metadata == ["key1:value1", "key2:value2"]

    def test_pass_none_for_required_parameters(self) -> None:
        """Test that passing None for any required parameter raises an error."""
        required_params = {
            "one_logger_training_telemetry_version": "one_logger_training_telemetry_version is required",
            "enable_for_current_rank": "enable_for_current_rank is required",
            "perf_tag": "perf_tag is required",
            "session_tag": "session_tag is required",
            "app_type": "app_type is required",
            "log_every_n_train_iterations": "log_every_n_train_iterations is required",
            "world_size": "world_size is required",
            "global_batch_size": "global_batch_size is required",
            "is_baseline_run": "is_baseline_run is required",
            "is_train_iterations_enabled": "is_train_iterations_enabled is required",
            "is_validation_iterations_enabled": "is_validation_iterations_enabled is required",
            "is_test_iterations_enabled": "is_test_iterations_enabled is required",
            "is_save_checkpoint_enabled": "is_save_checkpoint_enabled is required",
            "is_log_throughput_enabled": "is_log_throughput_enabled is required",
            "summary_data_schema_version": "summary_data_schema_version is required",
            "node_name": "node_name is required",
            "rank": "rank is required",
            "checkpoint_strategy": "checkpoint_strategy is required",
        }

        base_params = {
            "one_logger_training_telemetry_version": "1.0.0",
            "enable_for_current_rank": True,
            "perf_tag": "test_tag",
            "session_tag": "test_session",
            "app_type": "training",
            "log_every_n_train_iterations": 100,
            "world_size": 1,
            "global_batch_size": 32,
            "is_baseline_run": False,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "is_test_iterations_enabled": True,
            "is_save_checkpoint_enabled": True,
            "is_log_throughput_enabled": True,
            "summary_data_schema_version": "1.0",
            "node_name": "test_node",
            "rank": 0,
            "checkpoint_strategy": CheckPointStrategy.SYNC,
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
                TrainingMetricsUpdateAttributes.create(**params)


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
        )
        assert isinstance(attrs.checkpoint_strategy, CheckPointStrategy)
        assert attrs.checkpoint_strategy == CheckPointStrategy.SYNC
        assert attrs.current_iteration == 100
        assert attrs.first_successful_save_checkpoint_timestamp_sec == 1100.0
        assert attrs.latest_successful_save_checkpoint_timestamp_sec == 1100.0
        assert attrs.save_checkpoint_success_count == 5
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
            checkpoint_size=1000000,
            checkpoint_directory="/path/to/checkpoint",
            training_start_timestamp_sec=2000.0,
        )
        assert attrs.checkpoint_strategy == CheckPointStrategy.ASYNC
        assert attrs.current_iteration == 200
        assert attrs.first_successful_save_checkpoint_timestamp_sec == 2100.0
        assert attrs.latest_successful_save_checkpoint_timestamp_sec == 2100.0
        assert attrs.save_checkpoint_success_count == 5
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
        }

        base_params = {
            "checkpoint_strategy": CheckPointStrategy.SYNC,
            "current_iteration": 100,
            "first_successful_save_checkpoint_timestamp_sec": 1100.0,
            "latest_successful_save_checkpoint_timestamp_sec": 1100.0,
            "save_checkpoint_success_count": 2,
        }

        for param, error_msg in required_params.items():
            params = base_params.copy()
            params[param] = None  # type: ignore[assignment]
            with pytest.raises(OneLoggerError, match=error_msg):
                SaveCheckpointSuccessEventAttributes.create(**params)
