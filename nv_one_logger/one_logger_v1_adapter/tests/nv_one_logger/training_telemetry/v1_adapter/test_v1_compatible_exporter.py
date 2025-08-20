# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false

from unittest.mock import Mock

import pytest
from nv_one_logger.api.config import OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.internal.singleton import SingletonMeta
from nv_one_logger.core.span import StandardSpanName
from nv_one_logger.training_telemetry.api.attributes import TrainingTelemetryAttributes
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.events import StandardTrainingJobEventName
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter
from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import (
    V1CompatibleExporter,
    V1CompatibleWandbExporterAdapter,
    V1CompatibleWandbExporterAsync,
    V1CompatibleWandbExporterSync,
)


@pytest.fixture(autouse=True, scope="function")
def reset_v2_provider():
    """Reset the v2 provider singleton to isolate tests."""
    OneLoggerProvider.instance()._config = None  # type: ignore
    OneLoggerProvider.instance()._recorder = None  # type: ignore


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset the state of the singletons."""
    with SingletonMeta._lock:
        SingletonMeta._instances.pop(TrainingTelemetryProvider, None)
        SingletonMeta._instances.pop(OneLoggerProvider, None)


def create_test_one_logger_config(world_size: int = 8) -> OneLoggerConfig:
    """Create a test OneLoggerConfig instance."""
    return OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        world_size_or_fn=world_size,
        telemetry_config=TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        ),
    )


class TestV1CompatibleExporter:
    """Test cases for the V1CompatibleExporter factory class."""

    def test_factory_creates_sync_exporter(self, config: TrainingTelemetryConfig) -> None:
        """Test that factory creates V1CompatibleWandbExporterSync when async_mode=False."""
        # Create OneLoggerConfig with nested TrainingTelemetryConfig
        one_logger_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
            telemetry_config=config,
        )
        exporter_config = {"async_mode": False}
        exporter = V1CompatibleExporter(one_logger_config=one_logger_config, config=exporter_config)

        assert exporter.is_async is False
        assert isinstance(exporter.exporter, V1CompatibleWandbExporterSync)

    def test_factory_creates_async_exporter(self, config: TrainingTelemetryConfig) -> None:
        """Test that factory creates V1CompatibleWandbExporterAsync when async_mode=True."""
        # Create OneLoggerConfig with nested TrainingTelemetryConfig
        one_logger_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
            telemetry_config=config,
        )
        exporter_config = {"async_mode": True}
        exporter = V1CompatibleExporter(one_logger_config=one_logger_config, config=exporter_config)

        assert exporter.is_async is True
        assert isinstance(exporter.exporter, V1CompatibleWandbExporterAsync)

    def test_factory_default_creates_sync_exporter(self, config: TrainingTelemetryConfig) -> None:
        """Test that factory creates sync exporter by default."""
        # Create OneLoggerConfig with nested TrainingTelemetryConfig
        one_logger_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
            telemetry_config=config,
        )
        exporter_config = {}
        exporter = V1CompatibleExporter(one_logger_config=one_logger_config, config=exporter_config)

        assert exporter.is_async is False
        assert isinstance(exporter.exporter, V1CompatibleWandbExporterSync)


class TestV1CompatibleWandbExporterAdapterUpdateTrainingTelemetry:
    """Test cases for the _metrics_for_update_training_telemetry_event method."""

    def test_metrics_for_update_training_telemetry_event_success(self):
        """Test successful extraction of metrics for update training telemetry event."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span
        mock_span = Mock()
        mock_span.name = StandardSpanName.APPLICATION

        # Mock event
        mock_event = Mock()
        mock_event.name = StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG

        # Create training telemetry attributes
        attributes = TrainingTelemetryAttributes.create(
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
            custom_metadata=None,
        )
        mock_event.attributes = attributes

        # Mock perf_tag_dict method
        adapter._perf_tag_dict = Mock(
            return_value={
                "app_tag": ["test_perf"],
                "app_tag_id": ["hash_value"],
                "app_tag_count": 1,
            }
        )

        # Execute
        result = adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

        # Verify
        assert "global_batch_size" in result
        assert result["global_batch_size"] == 64
        assert "micro_batch_size" in result
        assert result["micro_batch_size"] == 32
        assert "model_seq_length" in result
        assert result["model_seq_length"] == 512
        assert "is_train_iterations_enabled" in result
        assert result["is_train_iterations_enabled"] is True
        assert "is_validation_iterations_enabled" in result
        assert result["is_validation_iterations_enabled"] is True
        assert "is_test_iterations_enabled" in result
        assert result["is_test_iterations_enabled"] is True
        assert "is_save_checkpoint_enabled" in result
        assert result["is_save_checkpoint_enabled"] is True
        assert "is_log_throughput_enabled" in result
        assert result["is_log_throughput_enabled"] is True
        assert "save_checkpoint_strategy" in result
        assert result["save_checkpoint_strategy"] == CheckPointStrategy.SYNC
        assert "train_iterations_target" in result
        assert result["train_iterations_target"] == 1000
        assert "train_samples_target" in result
        assert result["train_samples_target"] == 100000

        # Verify perf_tag_dict was called
        adapter._perf_tag_dict.assert_called_once_with(mock_config)

        # Verify perf_tag metrics are included
        assert "app_tag" in result
        assert result["app_tag"] == ["test_perf"]
        assert "app_tag_id" in result
        assert result["app_tag_id"] == ["hash_value"]
        assert "app_tag_count" in result
        assert result["app_tag_count"] == 1

    def test_metrics_for_update_training_telemetry_event_wrong_span_name(self):
        """Test that wrong span name raises an error."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span with wrong name
        mock_span = Mock()
        mock_span.name = "WRONG_SPAN_NAME"

        # Mock event
        mock_event = Mock()
        mock_event.name = StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG
        mock_event.attributes = Mock()

        # Execute and verify
        with pytest.raises(OneLoggerError, match="Expected span name to be APPLICATION"):
            adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

    def test_metrics_for_update_training_telemetry_event_wrong_event_name(self):
        """Test that wrong event name raises an error."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span
        mock_span = Mock()
        mock_span.name = StandardSpanName.APPLICATION

        # Mock event with wrong name
        mock_event = Mock()
        mock_event.name = "WRONG_EVENT_NAME"
        mock_event.attributes = Mock()

        # Execute and verify
        with pytest.raises(OneLoggerError, match="Expected event name to be UPDATE_TRAINING_TELEMETRY_CONFIG"):
            adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

    def test_metrics_for_update_training_telemetry_event_wrong_attributes_type(self):
        """Test that wrong attributes type raises an error."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span
        mock_span = Mock()
        mock_span.name = StandardSpanName.APPLICATION

        # Mock event with wrong attributes type
        mock_event = Mock()
        mock_event.name = StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG
        mock_event.attributes = Mock()  # Wrong type

        # Execute and verify
        with pytest.raises(OneLoggerError, match="Expected event attributes to be of type 'TrainingTelemetryAttributes'"):
            adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

    def test_metrics_for_update_training_telemetry_event_with_optional_fields_none(self):
        """Test handling of optional fields when they are None."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span
        mock_span = Mock()
        mock_span.name = StandardSpanName.APPLICATION

        # Mock event
        mock_event = Mock()
        mock_event.name = StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG

        # Create training telemetry attributes with optional fields as None
        attributes = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            # All optional fields are None by default
            custom_metadata=None,
        )
        mock_event.attributes = attributes

        # Mock perf_tag_dict method
        adapter._perf_tag_dict = Mock(
            return_value={
                "app_tag": ["test_perf"],
                "app_tag_id": ["hash_value"],
                "app_tag_count": 1,
            }
        )

        # Execute
        result = adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

        # Verify required fields are present
        assert "global_batch_size" in result
        assert result["global_batch_size"] == 64

        # Verify optional fields are omitted when None
        assert "micro_batch_size" not in result
        assert "model_seq_length" not in result
        assert "is_train_iterations_enabled" not in result
        assert "is_validation_iterations_enabled" not in result
        assert "is_test_iterations_enabled" not in result
        assert "is_save_checkpoint_enabled" not in result
        assert "is_log_throughput_enabled" not in result
        assert "save_checkpoint_strategy" not in result  # Should be omitted when None
        assert "train_iterations_target" not in result  # Should be omitted when None
        assert "train_samples_target" not in result  # Should be omitted when None

    def test_metrics_for_update_training_telemetry_event_save_checkpoint_disabled(self):
        """Test that save_checkpoint_strategy is None when save_checkpoint_enabled is False."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span
        mock_span = Mock()
        mock_span.name = StandardSpanName.APPLICATION

        # Mock event
        mock_event = Mock()
        mock_event.name = StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG

        # Create training telemetry attributes with save_checkpoint_enabled=False
        attributes = TrainingTelemetryAttributes.create(
            perf_tag="test_perf",
            global_batch_size=64,
            log_every_n_train_iterations=10,
            is_save_checkpoint_enabled=False,
            checkpoint_strategy=CheckPointStrategy.SYNC,  # This should be ignored
            custom_metadata=None,
        )
        mock_event.attributes = attributes

        # Mock perf_tag_dict method
        adapter._perf_tag_dict = Mock(
            return_value={
                "app_tag": ["test_perf"],
                "app_tag_id": ["hash_value"],
                "app_tag_count": 1,
            }
        )

        # Execute
        result = adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

        # Verify
        assert "is_save_checkpoint_enabled" in result
        assert result["is_save_checkpoint_enabled"] is False
        assert "save_checkpoint_strategy" not in result  # Should be omitted when is_save_checkpoint_enabled=False

    def test_metrics_for_update_training_telemetry_event_with_perf_tag_list(self):
        """Test handling of perf_tag as a list."""
        # Setup
        perf_tags = ["tag1", "tag2", "tag3"]
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn=perf_tags,
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Mock span
        mock_span = Mock()
        mock_span.name = StandardSpanName.APPLICATION

        # Mock event
        mock_event = Mock()
        mock_event.name = StandardTrainingJobEventName.UPDATE_TRAINING_TELEMETRY_CONFIG

        # Create training telemetry attributes
        attributes = TrainingTelemetryAttributes.create(
            perf_tag=perf_tags,
            global_batch_size=64,
            log_every_n_train_iterations=10,
            custom_metadata=None,
        )
        mock_event.attributes = attributes

        # Mock perf_tag_dict method
        adapter._perf_tag_dict = Mock(
            return_value={
                "app_tag": perf_tags,
                "app_tag_id": ["hash1", "hash2", "hash3"],
                "app_tag_count": 3,
            }
        )

        # Execute
        result = adapter._metrics_for_update_training_telemetry_event(mock_event, mock_span)

        # Verify perf_tag metrics are included
        assert "app_tag" in result
        assert result["app_tag"] == perf_tags
        assert "app_tag_id" in result
        assert result["app_tag_id"] == ["hash1", "hash2", "hash3"]
        assert "app_tag_count" in result
        assert result["app_tag_count"] == 3


class TestV1CompatibleWandbExporterAdapterPerfTagDict:
    """Test cases for the _perf_tag_dict method."""

    def test_perf_tag_dict_with_string_perf_tag(self):
        """Test _perf_tag_dict with string perf_tag."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Execute
        result = adapter._perf_tag_dict(mock_config)

        # Verify
        assert "app_tag" in result
        assert result["app_tag"] == ["test_perf"]
        assert "app_tag_id" in result
        assert len(result["app_tag_id"]) == 1
        assert "app_tag_count" in result
        assert result["app_tag_count"] == 1

    def test_perf_tag_dict_with_list_perf_tag(self):
        """Test _perf_tag_dict with list perf_tag."""
        # Setup
        perf_tags = ["tag1", "tag2", "tag3"]
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn=perf_tags,
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Execute
        result = adapter._perf_tag_dict(mock_config)

        # Verify
        assert "app_tag" in result
        assert result["app_tag"] == perf_tags
        assert "app_tag_id" in result
        assert len(result["app_tag_id"]) == 3
        assert "app_tag_count" in result
        assert result["app_tag_count"] == 3

    def test_perf_tag_dict_no_telemetry_config(self):
        """Test _perf_tag_dict when telemetry_config is None."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = None

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Execute and verify
        with pytest.raises(OneLoggerError, match="Training telemetry config is not set"):
            adapter._perf_tag_dict(mock_config)

    def test_perf_tag_dict_hash_consistency(self):
        """Test that _perf_tag_dict produces consistent hashes for the same perf_tag."""
        # Setup
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config)

        # Execute twice
        result1 = adapter._perf_tag_dict(mock_config)
        result2 = adapter._perf_tag_dict(mock_config)

        # Verify results are consistent
        assert result1["app_tag_id"] == result2["app_tag_id"]
        assert result1["app_tag"] == result2["app_tag"]
        assert result1["app_tag_count"] == result2["app_tag_count"]

    def test_perf_tag_dict_hash_uniqueness(self):
        """Test that _perf_tag_dict produces different hashes for different perf_tags."""
        # Setup
        mock_config1 = Mock(spec=OneLoggerConfig)
        mock_config1.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf1",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        mock_config2 = Mock(spec=OneLoggerConfig)
        mock_config2.telemetry_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf2",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        adapter = V1CompatibleWandbExporterAdapter(mock_config1)

        # Execute
        result1 = adapter._perf_tag_dict(mock_config1)
        result2 = adapter._perf_tag_dict(mock_config2)

        # Verify hashes are different
        assert result1["app_tag_id"] != result2["app_tag_id"]
        assert result1["app_tag"] != result2["app_tag"]
        assert result1["app_tag_count"] == result2["app_tag_count"] == 1


class TestV1CompatibleExporterUpdated:
    """Test cases for the updated V1CompatibleExporter class."""

    def test_v1_compatible_exporter_init_with_one_logger_config(self):
        """Test V1CompatibleExporter initialization with OneLoggerConfig."""
        # Setup
        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
        )

        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Set telemetry_config in base_config
        base_config.telemetry_config = training_config

        # Act
        exporter = V1CompatibleExporter(one_logger_config=base_config, config={"async_mode": False})

        # Assert
        assert exporter._one_logger_config == base_config
        assert exporter.is_async is False

    def test_v1_compatible_exporter_init_with_one_logger_config_async(self):
        """Test V1CompatibleExporter initialization with async mode."""
        # Setup
        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
        )

        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Set telemetry_config in base_config
        base_config.telemetry_config = training_config

        # Execute
        exporter = V1CompatibleExporter(one_logger_config=base_config, config={"async_mode": True})

        # Verify
        assert exporter._one_logger_config == base_config
        assert exporter.is_async is True

    def test_v1_compatible_exporter_run_name_generation(self):
        """Test that run_name is generated correctly with application_name."""
        # Setup
        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
        )

        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Set telemetry_config in base_config
        base_config.telemetry_config = training_config

        # Execute
        exporter = V1CompatibleExporter(one_logger_config=base_config, config={"async_mode": False})

        # Verify run_name format
        run_name = exporter._exporter_config.run_name
        assert run_name.startswith("test_app-run-")
        assert len(run_name) > len("test_app-run-")  # Should have UUID appended

    def test_v1_compatible_exporter_properties(self):
        """Test V1CompatibleExporter properties."""
        # Setup
        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
        )

        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Set telemetry_config in base_config
        base_config.telemetry_config = training_config

        # Execute
        exporter = V1CompatibleExporter(one_logger_config=base_config, config={"async_mode": False})

        # Verify properties
        assert exporter._one_logger_config == base_config
        assert exporter._exporter_config is not None
        assert exporter.is_async is False
        assert exporter.exporter is not None

    def test_v1_compatible_exporter_async_properties(self):
        """Test V1CompatibleExporter properties in async mode."""
        # Setup
        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=8,
        )

        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Set telemetry_config in base_config
        base_config.telemetry_config = training_config

        # Execute
        exporter = V1CompatibleExporter(one_logger_config=base_config, config={"async_mode": True})

        # Verify properties
        assert exporter._one_logger_config == base_config
        assert exporter._exporter_config is not None
        assert exporter.is_async is True
        assert exporter.exporter is not None


class TestConfigAdapterUpdated:
    """Test cases for the updated ConfigAdapter class."""

    def test_convert_to_v2_config_returns_one_logger_config(self):
        """Test that convert_to_v2_config returns OneLoggerConfig instead of TrainingTelemetryConfig."""
        # Setup
        v1_config = {
            "one_logger_project": "test_project",
            "app_tag": "test_perf",
            "app_tag_run_name": "test_session",
            "world_size": 8,
            "global_batch_size": 64,
            "log_every_n_train_iterations": 10,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "is_test_iterations_enabled": True,
            "is_save_checkpoint_enabled": True,
            "is_log_throughput_enabled": False,
            "save_checkpoint_strategy": "sync",
            "micro_batch_size": 32,
            "flops_per_sample": 1000,
            "train_iterations_target": 1000,
            "train_samples_target": 100000,
            "enable_for_current_rank": True,
            "quiet": False,
        }

        # Execute
        base_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Verify return types
        assert isinstance(base_config, OneLoggerConfig)
        assert base_config.telemetry_config is not None
        assert isinstance(base_config.telemetry_config, TrainingTelemetryConfig)

        # Verify base config fields
        assert base_config.application_name == "test_project"
        assert base_config.session_tag == "test_session"
        assert base_config.enable_for_current_rank is True

        # Verify telemetry config fields
        telemetry_config = base_config.telemetry_config
        assert telemetry_config.perf_tag == "test_perf"
        assert base_config.world_size == 8
        assert telemetry_config.global_batch_size == 64
        assert telemetry_config.log_every_n_train_iterations == 10
        assert telemetry_config.is_train_iterations_enabled is True
        assert telemetry_config.is_validation_iterations_enabled is True
        assert telemetry_config.is_test_iterations_enabled is True
        assert telemetry_config.is_save_checkpoint_enabled is True
        assert telemetry_config.is_log_throughput_enabled is False
        assert telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC
        assert telemetry_config.micro_batch_size == 32
        assert telemetry_config.flops_per_sample == 1000
        assert telemetry_config.train_iterations_target == 1000
        assert telemetry_config.train_samples_target == 100000

    def test_convert_to_v2_config_with_optional_fields(self):
        """Test convert_to_v2_config with optional fields."""
        # Setup
        v1_config = {
            "one_logger_project": "test_project",
            "app_tag": "test_perf",
            "app_tag_run_name": "test_session",
            "world_size": 8,
            "global_batch_size": 64,
            "log_every_n_train_iterations": 10,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            # Optional fields not provided
        }

        # Execute
        base_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Verify
        assert isinstance(base_config, OneLoggerConfig)
        assert base_config.telemetry_config is not None

        telemetry_config = base_config.telemetry_config
        # Required fields should be present
        assert telemetry_config.perf_tag == "test_perf"
        assert base_config.world_size == 8
        assert telemetry_config.global_batch_size == 64
        assert telemetry_config.log_every_n_train_iterations == 10
        assert telemetry_config.is_train_iterations_enabled is True
        assert telemetry_config.is_validation_iterations_enabled is True

        # Optional fields should have default values
        assert telemetry_config.is_test_iterations_enabled is True  # Default
        assert telemetry_config.is_save_checkpoint_enabled is True  # Default
        assert telemetry_config.is_log_throughput_enabled is False  # Default
        assert telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC  # Default
        assert telemetry_config.micro_batch_size is None
        assert telemetry_config.flops_per_sample is None
        assert telemetry_config.train_iterations_target is None
        assert telemetry_config.train_samples_target is None

    def test_convert_to_v2_config_with_async_checkpoint_strategy(self):
        """Test convert_to_v2_config with async checkpoint strategy."""
        # Setup
        v1_config = {
            "one_logger_project": "test_project",
            "app_tag": "test_perf",
            "app_tag_run_name": "test_session",
            "world_size": 8,
            "global_batch_size": 64,
            "log_every_n_train_iterations": 10,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "save_checkpoint_strategy": "async",
        }

        # Execute
        base_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Verify
        assert base_config.telemetry_config.save_checkpoint_strategy == CheckPointStrategy.ASYNC

    def test_convert_to_v2_config_with_metadata(self):
        """Test convert_to_v2_config with metadata."""
        # Setup
        metadata = {"key1": "value1", "key2": "value2"}
        v1_config = {
            "one_logger_project": "test_project",
            "app_tag": "test_perf",
            "app_tag_run_name": "test_session",
            "world_size": 8,
            "global_batch_size": 64,
            "log_every_n_train_iterations": 10,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "metadata": metadata,
        }

        # Execute
        base_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Verify
        assert base_config.custom_metadata == metadata

    def test_convert_to_v2_config_with_quiet_mode(self):
        """Test convert_to_v2_config with quiet mode."""
        # Setup
        v1_config = {
            "one_logger_project": "test_project",
            "app_tag": "test_perf",
            "app_tag_run_name": "test_session",
            "world_size": 8,
            "global_batch_size": 64,
            "log_every_n_train_iterations": 10,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "quiet": True,
        }

        # Execute
        base_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Verify
        assert base_config.error_handling_strategy == OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR

    def test_convert_to_v2_config_wandb_config(self):
        """Test that wandb_config is created correctly."""
        # Setup
        v1_config = {
            "one_logger_project": "test_project",
            "app_tag": "test_perf",
            "app_tag_run_name": "test_session",
            "world_size": 8,
            "global_batch_size": 64,
            "log_every_n_train_iterations": 10,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "quiet": False,  # Explicitly set to False, so should get PROPAGATE_EXCEPTIONS
        }

        # Execute
        base_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Verify error handling strategy matches the config
        assert base_config.error_handling_strategy == OneLoggerErrorHandlingStrategy.PROPAGATE_EXCEPTIONS
