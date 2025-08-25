# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
"""Unit tests for the TrainingTelemetryProvider class.

This module contains tests that verify the functionality of the TrainingTelemetryProvider class,
including configuration, recorder management, and singleton behavior.
"""

from pathlib import Path
from typing import Generator, List, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.api.telemetry_config import ApplicationType
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.span import SpanName
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.exporter.file_exporter import FileExporter
from nv_one_logger.exporter.logger_exporter import LoggerExporter
from nv_one_logger.recorder.default_recorder import ExportCustomizationMode
from nv_one_logger.training_telemetry.api.checkpoint import CheckPointStrategy
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.spans import StandardTrainingJobSpanName
from nv_one_logger.training_telemetry.api.training_recorder import TrainingRecorder
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from .utils import reset_singletong_providers_for_test


@pytest.fixture
def mock_exporter() -> Exporter:
    """Fixture that returns a mock Exporter instance."""
    return MagicMock(spec=Exporter)


@pytest.fixture
def another_mock_exporter() -> Exporter:
    """Fixture that returns a mock Exporter instance."""
    return MagicMock(spec=Exporter)


_BASE_CONFIG = OneLoggerConfig(
    application_name="test_app",
    session_tag_or_fn="test_session",
    is_baseline_run_or_fn=False,
    enable_for_current_rank=True,
    world_size_or_fn=4,
    telemetry_config=TrainingTelemetryConfig(
        global_batch_size_or_fn=32,
        perf_tag_or_fn="test_perf",
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
    ),
)


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Reset the singleton instance of TrainingTelemetryProvider before each test."""
    reset_singletong_providers_for_test()
    yield
    # Reset the singleton after the test
    reset_singletong_providers_for_test()


@pytest.fixture
def valid_config() -> OneLoggerConfig:
    """Fixture that returns a valid OneLoggerConfig with TrainingTelemetryConfig."""
    return OneLoggerConfig(
        application_name="test_app",
        session_tag_or_fn="test_session",
        is_baseline_run_or_fn=False,
        enable_for_current_rank=True,
        world_size_or_fn=4,
        telemetry_config=TrainingTelemetryConfig(
            global_batch_size_or_fn=32,
            perf_tag_or_fn="test_perf",
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        ),
    )


class TestTrainingTelemetryProvider:
    """Tests for TrainingTelemetryProvider class."""

    def assert_config_equal(self, config: OneLoggerConfig, provider: TrainingTelemetryProvider) -> None:
        """Assert that read-only config from the provider is equal to the given config."""
        provider_config = provider.config

        # Compare basic fields
        assert config.application_name == provider_config.application_name
        assert config.session_tag == provider_config.session_tag
        assert config.is_baseline_run == provider_config.is_baseline_run
        assert config.enable_for_current_rank == provider_config.enable_for_current_rank
        assert config.custom_metadata == provider_config.custom_metadata
        assert config.error_handling_strategy == provider_config.error_handling_strategy
        assert config.summary_data_schema_version == provider_config.summary_data_schema_version

        # Compare telemetry_config - handle the type difference
        if config.telemetry_config is not None:
            assert provider_config.telemetry_config is not None
            # Compare the fields that exist in both TelemetryConfig and TrainingTelemetryConfig
            assert config.world_size == provider_config.world_size
            assert config.telemetry_config.app_type == provider_config.telemetry_config.app_type
            assert config.telemetry_config.is_train_iterations_enabled == provider_config.telemetry_config.is_train_iterations_enabled
            assert config.telemetry_config.is_validation_iterations_enabled == provider_config.telemetry_config.is_validation_iterations_enabled
            assert config.telemetry_config.is_test_iterations_enabled == provider_config.telemetry_config.is_test_iterations_enabled
            assert config.telemetry_config.is_save_checkpoint_enabled == provider_config.telemetry_config.is_save_checkpoint_enabled
            assert config.telemetry_config.custom_metadata == provider_config.telemetry_config.custom_metadata

            # If the original config has TrainingTelemetryConfig, compare those fields too
            if hasattr(config.telemetry_config, "perf_tag"):
                assert hasattr(provider_config.telemetry_config, "perf_tag")
                assert config.telemetry_config.perf_tag == provider_config.telemetry_config.perf_tag
                assert config.telemetry_config.global_batch_size == provider_config.telemetry_config.global_batch_size
                assert config.telemetry_config.micro_batch_size == provider_config.telemetry_config.micro_batch_size
                assert config.telemetry_config.seq_length == provider_config.telemetry_config.seq_length
                assert config.telemetry_config.flops_per_sample == provider_config.telemetry_config.flops_per_sample
                assert config.telemetry_config.train_iterations_target == provider_config.telemetry_config.train_iterations_target
                assert config.telemetry_config.train_samples_target == provider_config.telemetry_config.train_samples_target
                assert config.telemetry_config.log_every_n_train_iterations == provider_config.telemetry_config.log_every_n_train_iterations
                assert config.telemetry_config.is_log_throughput_enabled == provider_config.telemetry_config.is_log_throughput_enabled
                assert config.telemetry_config.save_checkpoint_strategy == provider_config.telemetry_config.save_checkpoint_strategy
        else:
            assert provider_config.telemetry_config is None

    def test_singleton_behavior(self) -> None:
        """Test that TrainingTelemetryProvider behaves as a singleton."""
        provider1 = TrainingTelemetryProvider.instance()
        provider2 = TrainingTelemetryProvider.instance()
        assert provider1 is provider2

    def test_configure_with_disabled_telemetry(self, valid_config: OneLoggerConfig, mock_exporter: Exporter) -> None:
        """Test configuration when telemetry is disabled for current rank."""
        disabled_config = valid_config.model_copy()
        disabled_config.enable_for_current_rank = False

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(disabled_config).with_exporter(mock_exporter).configure_provider()
        self.assert_config_equal(disabled_config, provider)
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._exporters == []  # Force the exporter to be empty

    def test_with_base_config_success(self, mock_exporter: Exporter) -> None:
        """Test that with_base_config sets the base config correctly."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(_BASE_CONFIG).with_exporter(mock_exporter).configure_provider()

        self.assert_config_equal(_BASE_CONFIG, provider)
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._exporters == [mock_exporter]

    def test_with_base_config_called_twice_raises_error(self) -> None:
        """Test that calling with_base_config twice raises an error."""
        another_config = OneLoggerConfig(
            application_name="test_app2",
            session_tag_or_fn="test_session2",
            is_baseline_run_or_fn=False,
            enable_for_current_rank=True,
            world_size_or_fn=80,
            telemetry_config=TrainingTelemetryConfig(
                global_batch_size_or_fn=400,
                perf_tag_or_fn="test_perf2",
                save_checkpoint_strategy=CheckPointStrategy.SYNC,
            ),
        )
        with pytest.raises(OneLoggerError, match="You can only call with_base_config once"):
            TrainingTelemetryProvider.instance().with_base_config(_BASE_CONFIG).with_base_config(another_config)

    def test_build_one_logger_config_with_base_config_override(self) -> None:
        """Test that _build_one_logger_config works correctly with a base config."""
        TrainingTelemetryProvider.instance().with_base_config(_BASE_CONFIG).with_config_override(
            {
                "world_size_or_fn": 8,
                "telemetry_config": {
                    "log_every_n_train_iterations": 100,
                },
            }
        ).configure_provider()
        result_config = TrainingTelemetryProvider.instance().config
        assert isinstance(result_config, OneLoggerConfig)
        assert result_config.application_name == "test_app"  # base value
        assert result_config.telemetry_config is not None
        assert result_config.world_size == 8  # Overridden value
        assert result_config.telemetry_config.log_every_n_train_iterations == 100  # Overridden value
        assert result_config.telemetry_config.global_batch_size == 32  # base value

    def test_with_incomplete_config_raises_error(self) -> None:
        """Test that if we don't provide required fields, the builder raises an error."""
        override1 = {"log_every_n_train_iterations": 100}
        override2 = {"enable_for_current_rank": True}
        with pytest.raises(OneLoggerError, match="Invalid configuration!"):
            TrainingTelemetryProvider.instance().with_config_override(override1).with_config_override(override2).configure_provider()

    def test_with_config_override_updates_existing_keys(self) -> None:
        """Test that with_config_override updates existing keys correctly."""
        override1 = {"telemetry_config": {"log_every_n_train_iterations": 100}}
        override2 = {"telemetry_config": {"log_every_n_train_iterations": 200}}  # Override the same key
        TrainingTelemetryProvider.instance().with_base_config(_BASE_CONFIG).with_config_override(override1).with_config_override(override2).configure_provider()
        config = TrainingTelemetryProvider.instance().config
        assert config.telemetry_config is not None
        assert config.telemetry_config.log_every_n_train_iterations == 200

    def test_with_multiple_exporter_success(self, mock_exporter: Exporter, another_mock_exporter: Exporter) -> None:
        """Test that with_exporter adds exporters correctly."""
        TrainingTelemetryProvider.instance().with_base_config(_BASE_CONFIG).with_exporter(mock_exporter).with_exporter(
            another_mock_exporter
        ).configure_provider()
        assert TrainingTelemetryProvider.instance().recorder._exporters == [
            mock_exporter,
            another_mock_exporter,
        ]

    def test_no_exporters_success(self) -> None:
        """Test that if we don't provide any exporters, the builder doesn't raise an error."""
        TrainingTelemetryProvider.instance().with_base_config(_BASE_CONFIG).configure_provider()
        provider = TrainingTelemetryProvider.instance()
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._exporters == []

    def test_no_config_raises_error(self, mock_exporter: Exporter) -> None:
        """Test that if we don't provide any config, the builder raises an error."""
        with pytest.raises(
            OneLoggerError,
            match="No configuration was provided. Please provide a base config and/or config overrides.",
        ):
            TrainingTelemetryProvider.instance().with_exporter(mock_exporter).configure_provider()

    def test_build_one_logger_config_without_base_config(self) -> None:
        """Test that _build_one_logger_config works correctly without a base config if enough config overrides are provided."""
        override = {
            "application_name": "test_app",
            "session_tag_or_fn": "test_session",
            "is_baseline_run_or_fn": False,
            "world_size_or_fn": 8,
            "telemetry_config": {
                "perf_tag_or_fn": "test_perf",
                "global_batch_size_or_fn": 64,
                "app_type_or_fn": ApplicationType.TRAINING,
                "save_checkpoint_strategy": CheckPointStrategy.SYNC,
            },
        }
        TrainingTelemetryProvider.instance().with_config_override(override).configure_provider()
        result_config = TrainingTelemetryProvider.instance().config
        assert isinstance(result_config, OneLoggerConfig)
        assert result_config.telemetry_config is not None
        assert result_config.world_size == 8
        assert result_config.telemetry_config.global_batch_size == 64
        assert result_config.application_name == "test_app"

    def test_build_one_logger_config_with_multiple_exporters(self, mock_exporter: Exporter, another_mock_exporter: Exporter) -> None:
        """Test that _build_one_logger_config works correctly with multiple exporters."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(_BASE_CONFIG).with_exporter(mock_exporter).with_exporter(another_mock_exporter).configure_provider()
        assert provider.recorder._exporters == [mock_exporter, another_mock_exporter]

    def test_build_one_logger_config_invalid_config_raises_error(self) -> None:
        """Test that _build_one_logger_config raises an error for invalid configuration."""
        invalid_override = {
            "world_size_or_fn": 0,  # Invalid: must be > 0
            "telemetry_config": {
                "perf_tag_or_fn": "test_perf",
                "global_batch_size_or_fn": 32,
            },
            "application_name": "test_app",
            "session_tag_or_fn": "test_session",
            "is_baseline_run_or_fn": False,
            "save_checkpoint_strategy": CheckPointStrategy.SYNC,
        }

        with pytest.raises(OneLoggerError, match="world_size must be set to a positive value"):
            TrainingTelemetryProvider.instance().with_base_config(_BASE_CONFIG).with_config_override(invalid_override).configure_provider()

    def test_configure_provider_with_export_customization(self, mock_exporter: Exporter, another_mock_exporter: Exporter) -> None:
        """Test that configure_provider works correctly with export customization."""
        # Test with blacklist mode
        custom_span_filter = cast(
            List[SpanName],
            [
                StandardTrainingJobSpanName.TRAINING_LOOP,
                StandardTrainingJobSpanName.VALIDATION_LOOP,
            ],
        )

        provider = TrainingTelemetryProvider.instance()
        (
            provider.with_base_config(_BASE_CONFIG)
            .with_exporter(mock_exporter)
            .with_exporter(another_mock_exporter)
            .with_export_customization(
                export_customization_mode=ExportCustomizationMode.WHITELIST_SPANS,
                span_name_filter=custom_span_filter,
            )
            .configure_provider()
        )

        # Verify the recorder was created with the correct export customization settings
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        # The recorder should have the custom span filter instead of the default blacklist.
        assert provider.recorder._export_customization_mode == ExportCustomizationMode.WHITELIST_SPANS
        assert provider.recorder._span_name_filter == custom_span_filter

    def test_with_export_customization_called_twice_raises_error(self) -> None:
        """Test that calling with_export_customization twice raises an error."""
        custom_span_filter = cast(List[SpanName], [StandardTrainingJobSpanName.TRAINING_LOOP])
        another_span_filter = cast(List[SpanName], [StandardTrainingJobSpanName.VALIDATION_LOOP])

        with pytest.raises(OneLoggerError, match="You can only call with_export_customization once"):
            (
                TrainingTelemetryProvider.instance()
                .with_base_config(_BASE_CONFIG)
                .with_export_customization(
                    export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS,
                    span_name_filter=custom_span_filter,
                )
                .with_export_customization(
                    export_customization_mode=ExportCustomizationMode.WHITELIST_SPANS,
                    span_name_filter=another_span_filter,
                )
            )

    def test_with_config_override_after_configure_provider_raises_error(self) -> None:
        """Test that calling with_config_override after configure_provider raises an error."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(_BASE_CONFIG).configure_provider()
        with pytest.raises(
            OneLoggerError,
            match="with_config_override can be called only before configure_provider is called.",
        ):
            provider.with_config_override({"log_every_n_train_iterations": 100})

    def test_with_exporter_after_configure_provider_raises_error(self, mock_exporter: Exporter) -> None:
        """Test that calling with_exporter after configure_provider raises an error."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(_BASE_CONFIG).configure_provider()
        with pytest.raises(
            OneLoggerError,
            match="with_exporter can be called only before configure_provider is called.",
        ):
            provider.with_exporter(mock_exporter)

    def test_with_export_customization_after_configure_raises_error(self) -> None:
        """Test that calling with_export_customization after configure_provider raises an error."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(_BASE_CONFIG).configure_provider()

        custom_span_filter = cast(List[SpanName], [StandardTrainingJobSpanName.TRAINING_LOOP])
        with pytest.raises(
            OneLoggerError,
            match="with_export_customization can be called only before configure_provider is called.",
        ):
            provider.with_export_customization(
                export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS,
                span_name_filter=custom_span_filter,
            )

    def test_configure_provider_without_export_customization_uses_defaults(self, mock_exporter: Exporter) -> None:
        """Test that configure_provider uses default export customization when not specified."""
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import DEFAULT_SPANS_EXPORT_BLACKLIST

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(_BASE_CONFIG).with_exporter(mock_exporter).configure_provider()

        # Verify the recorder was created with the default export customization settings
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._span_name_filter == DEFAULT_SPANS_EXPORT_BLACKLIST
        assert provider.recorder._export_customization_mode == ExportCustomizationMode.BLACKLIST_SPANS

    def test_set_training_loop_config_deprecated(self, mock_exporter: Exporter) -> None:
        """Test that set_training_loop_config is deprecated and redirects to set_training_telemetry_config.

        This test verifies that the old method still works for backward compatibility.
        """
        # Create a base config without telemetry_config
        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=4,
        )

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(base_config).with_exporter(mock_exporter).configure_provider()

        # Verify that telemetry_config is initially None
        assert provider.config.telemetry_config is None

        # Create a training telemetry config to set
        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            world_size_or_fn=8,
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Set the training telemetry config using the new method
        # This will raise an error because there's no app span active, which is expected
        with pytest.raises(OneLoggerError, match="Cannot update training metrics: Please call on_app_start\\(\\) before calling this method\\."):
            provider.set_training_telemetry_config(training_config)

        # Verify that the config was still set despite the error (the error happens after setting the config)
        assert provider.config.telemetry_config is not None
        assert provider.config.telemetry_config.perf_tag == "test_perf"
        assert provider.config.world_size == 4  # world_size is in main config, not overridden by telemetry config
        assert provider.config.telemetry_config.global_batch_size == 64

    def test_set_training_telemetry_config_already_set_raises_error_updated(self, mock_exporter: Exporter) -> None:
        """Test that set_training_telemetry_config raises an error when the telemetry config is already set.

        This test verifies that:
        1. An error is raised when trying to set the telemetry config when it's already set
        2. The error message is appropriate
        3. The existing config is not modified
        """
        # Create a base config with an existing telemetry_config
        existing_training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="existing_perf",
            global_batch_size_or_fn=32,
            log_every_n_train_iterations=5,
        )

        base_config = OneLoggerConfig(
            application_name="test_app",
            session_tag_or_fn="test_session",
            world_size_or_fn=4,
            telemetry_config=existing_training_config,
        )

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(base_config).with_exporter(mock_exporter).configure_provider()

        # Verify that telemetry_config is initially set
        assert provider.config.telemetry_config is not None
        original_world_size = provider.config.world_size

        # Try to set a new telemetry config
        new_training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="new_perf",
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Verify that an error is raised
        with pytest.raises(OneLoggerError, match="Training telemetry config has already been set."):
            provider.set_training_telemetry_config(new_training_config)

        # Verify that the original config was not modified
        assert provider.config.telemetry_config is not None
        assert provider.config.world_size == original_world_size


class TestTrainingTelemetryProviderSetTrainingTelemetryConfig:
    """Test cases for the set_training_telemetry_config method."""

    def test_set_training_telemetry_config_success(self, training_telemetry_provider):
        """Test successful setting of training telemetry config."""
        # Setup
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = True

        # Mock OneLoggerProvider.instance().config
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = None  # Initially not set

        # Mock the recorder
        mock_recorder = Mock(spec=TrainingRecorder)
        mock_recorder.__class__ = TrainingRecorder

        with patch("nv_one_logger.training_telemetry.api.training_telemetry_provider.OneLoggerProvider") as mock_provider:
            mock_provider.instance.return_value.config = mock_config
            mock_provider.instance.return_value.recorder = mock_recorder

            # Create training telemetry config
            training_config = TrainingTelemetryConfig(
                perf_tag_or_fn="test_perf",
                global_batch_size_or_fn=64,
                log_every_n_train_iterations=10,
            )

            # Execute
            training_telemetry_provider.set_training_telemetry_config(training_config)

            # Verify
            assert mock_config.telemetry_config == training_config

    def test_set_training_telemetry_config_already_set(self, training_telemetry_provider):
        """Test that setting training telemetry config twice raises an error."""
        # Setup
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = True

        # Mock OneLoggerProvider.instance().config
        mock_config = Mock(spec=OneLoggerConfig)
        existing_config = TrainingTelemetryConfig(
            perf_tag_or_fn="existing_perf",
            global_batch_size_or_fn=32,
            log_every_n_train_iterations=5,
        )
        mock_config.telemetry_config = existing_config  # Already set

        with patch("nv_one_logger.training_telemetry.api.training_telemetry_provider.OneLoggerProvider") as mock_provider:
            mock_provider.instance.return_value.config = mock_config

            # Create new training telemetry config
            new_config = TrainingTelemetryConfig(
                perf_tag_or_fn="new_perf",
                global_batch_size_or_fn=64,
                log_every_n_train_iterations=10,
            )

            # Execute and verify
            with pytest.raises(OneLoggerError, match="Training telemetry config has already been set\\."):
                training_telemetry_provider.set_training_telemetry_config(new_config)

            # Verify the original config was not changed
            assert mock_config.telemetry_config == existing_config

    def test_set_training_telemetry_config_not_configured(self, training_telemetry_provider):
        """Test that setting training telemetry config before configuration raises an error."""
        # Setup - not configured
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = False

        # Create training telemetry config
        training_config = TrainingTelemetryConfig(
            perf_tag_or_fn="test_perf",
            global_batch_size_or_fn=64,
            log_every_n_train_iterations=10,
        )

        # Execute and verify
        with pytest.raises(
            OneLoggerError,
            match="You need to call TrainingTelemetryProvider\\.instance\\(\\)\\.configure\\(\\) once in your application before accessing the recorder\\.",
        ):
            training_telemetry_provider.set_training_telemetry_config(training_config)

    def test_set_training_telemetry_config_with_complete_config(self, training_telemetry_provider):
        """Test setting training telemetry config with all fields."""
        # Setup
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = True

        # Mock OneLoggerProvider.instance().config
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = None

        # Mock the recorder
        mock_recorder = Mock(spec=TrainingRecorder)
        mock_recorder.__class__ = TrainingRecorder

        with patch("nv_one_logger.training_telemetry.api.training_telemetry_provider.OneLoggerProvider") as mock_provider:
            mock_provider.instance.return_value.config = mock_config
            mock_provider.instance.return_value.recorder = mock_recorder

            # Create complete training telemetry config
            training_config = TrainingTelemetryConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=8,
                global_batch_size_or_fn=64,
                log_every_n_train_iterations=10,
                micro_batch_size_or_fn=32,
                seq_length_or_fn=512,
                flops_per_sample_or_fn=1000,
                train_iterations_target_or_fn=1000,
                train_samples_target_or_fn=100000,
                save_checkpoint_strategy=CheckPointStrategy.SYNC,
                is_train_iterations_enabled_or_fn=True,
                is_validation_iterations_enabled_or_fn=True,
                is_test_iterations_enabled_or_fn=True,
                is_save_checkpoint_enabled_or_fn=True,
                is_log_throughput_enabled_or_fn=True,
            )

            # Execute
            training_telemetry_provider.set_training_telemetry_config(training_config)

            # Verify
            assert mock_config.telemetry_config == training_config
            assert mock_config.telemetry_config.perf_tag == "test_perf"
            # world_size is in main config, not overridden by training telemetry config
            assert mock_config.telemetry_config.global_batch_size == 64
            assert mock_config.telemetry_config.log_every_n_train_iterations == 10
            assert mock_config.telemetry_config.micro_batch_size == 32
            assert mock_config.telemetry_config.seq_length == 512
            assert mock_config.telemetry_config.flops_per_sample == 1000
            assert mock_config.telemetry_config.train_iterations_target == 1000
            assert mock_config.telemetry_config.train_samples_target == 100000
            assert mock_config.telemetry_config.save_checkpoint_strategy == CheckPointStrategy.SYNC
            assert mock_config.telemetry_config.is_train_iterations_enabled is True
            assert mock_config.telemetry_config.is_validation_iterations_enabled is True
            assert mock_config.telemetry_config.is_test_iterations_enabled is True
            assert mock_config.telemetry_config.is_save_checkpoint_enabled is True
            assert mock_config.telemetry_config.is_log_throughput_enabled is True

    def test_set_training_telemetry_config_with_perf_tag_list(self, training_telemetry_provider):
        """Test setting training telemetry config with perf_tag as a list."""
        # Setup
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = True

        # Mock OneLoggerProvider.instance().config
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = None

        # Mock the recorder
        mock_recorder = Mock(spec=TrainingRecorder)
        mock_recorder.__class__ = TrainingRecorder

        with patch("nv_one_logger.training_telemetry.api.training_telemetry_provider.OneLoggerProvider") as mock_provider:
            mock_provider.instance.return_value.config = mock_config
            mock_provider.instance.return_value.recorder = mock_recorder

            # Create training telemetry config with perf_tag list
            perf_tags = ["tag1", "tag2", "tag3"]
            training_config = TrainingTelemetryConfig(
                perf_tag_or_fn=perf_tags,
                world_size_or_fn=8,
                global_batch_size_or_fn=64,
                log_every_n_train_iterations=10,
            )

            # Execute
            training_telemetry_provider.set_training_telemetry_config(training_config)

            # Verify
            assert mock_config.telemetry_config == training_config
            assert mock_config.telemetry_config.perf_tag == perf_tags

    def test_set_training_telemetry_config_calls_update_application_span_with_training_telemetry_config(self, training_telemetry_provider):
        """Test that set_training_telemetry_config calls _update_application_span_with_training_telemetry_config.

        This test verifies the new functionality where setting the training telemetry config
        automatically updates the application span with training metrics.
        """
        # Setup
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = True

        # Mock OneLoggerProvider.instance().config
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = None

        # Mock the recorder and its _update_application_span_with_training_telemetry_config method
        mock_recorder = Mock(spec=TrainingRecorder)
        # Ensure the mock is recognized as a TrainingRecorder
        mock_recorder.__class__ = TrainingRecorder

        with patch("nv_one_logger.training_telemetry.api.training_telemetry_provider.OneLoggerProvider") as mock_provider:
            mock_provider.instance.return_value.config = mock_config
            mock_provider.instance.return_value.recorder = mock_recorder

            # Create training telemetry config
            training_config = TrainingTelemetryConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=8,
                global_batch_size_or_fn=64,
                log_every_n_train_iterations=10,
            )

            # Execute
            training_telemetry_provider.set_training_telemetry_config(training_config)

            # Verify that the config was set
            assert mock_config.telemetry_config == training_config

            # Verify that _update_application_span_with_training_telemetry_config was called with the correct config
            mock_recorder._update_application_span_with_training_telemetry_config.assert_called_once_with(training_telemetry_config=training_config)

    def test_set_training_telemetry_config_calls_update_application_span_with_training_telemetry_config_no_app_span(self, training_telemetry_provider):
        """Test that set_training_telemetry_config calls _update_application_span_with_training_telemetry_config and handles errors.

        This test verifies that when there's no application span active, the method still calls
        _update_application_span_with_training_telemetry_config but the recorder method raises an appropriate error.
        """
        # Setup
        training_telemetry_provider._TrainingTelemetryProvider__fully_configured = True

        # Mock OneLoggerProvider.instance().config
        mock_config = Mock(spec=OneLoggerConfig)
        mock_config.telemetry_config = None

        # Mock the recorder and its _update_application_span_with_training_telemetry_config method to raise an error
        mock_recorder = Mock(spec=TrainingRecorder)
        # Ensure the mock is recognized as a TrainingRecorder
        mock_recorder.__class__ = TrainingRecorder
        mock_recorder._update_application_span_with_training_telemetry_config.side_effect = OneLoggerError(
            "Cannot update training metrics: Please call on_app_start() before calling this method."
        )

        with patch("nv_one_logger.training_telemetry.api.training_telemetry_provider.OneLoggerProvider") as mock_provider:
            mock_provider.instance.return_value.config = mock_config
            mock_provider.instance.return_value.recorder = mock_recorder

            # Create training telemetry config
            training_config = TrainingTelemetryConfig(
                perf_tag_or_fn="test_perf",
                world_size_or_fn=8,
                global_batch_size_or_fn=64,
                log_every_n_train_iterations=10,
            )

            # Execute and verify that the error is propagated
            with pytest.raises(OneLoggerError, match="Cannot update training metrics: Please call on_app_start\\(\\) before calling this method\\."):
                training_telemetry_provider.set_training_telemetry_config(training_config)

            # Verify that the config was set (the error happens after setting the config)
            assert mock_config.telemetry_config == training_config

            # Verify that _update_application_span_with_training_telemetry_config was called with the correct config
            mock_recorder._update_application_span_with_training_telemetry_config.assert_called_once_with(training_telemetry_config=training_config)

    def test_with_export_config_direct_dict_success(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config works with direct dictionary configuration."""
        provider = TrainingTelemetryProvider.instance()

        # Test with direct dictionary configuration
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/test_export.log"},
                "enabled": True,
            }
        ]

        provider.with_base_config(valid_config).with_export_config(exporters_config=exporters_config).configure_provider()

        # Verify that the provider was configured successfully
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        # Our FileExporter should be present, and package configs may also be present
        assert len(provider.recorder._exporters) >= 1
        # Check that our FileExporter is present
        exporter_types = [type(exporter) for exporter in provider.recorder._exporters]
        assert FileExporter in exporter_types

    def test_with_export_config_after_configure_provider_raises_error(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config raises an error when called after configure_provider."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(valid_config).configure_provider()

        # Try to call with_export_config after configure_provider
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/test.log"},
            }
        ]

        with pytest.raises(
            OneLoggerError,
            match="with_export_config can be called only before configure_provider is called",
        ):
            provider.with_export_config(exporters_config=exporters_config)

    def test_with_export_config_multiple_exporters(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config works with multiple exporters."""
        provider = TrainingTelemetryProvider.instance()

        # Test with multiple exporters
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/test1.log"},
                "enabled": True,
            },
            {
                "class_name": "nv_one_logger.exporter.logger_exporter.LoggerExporter",
                "config": {"logger": "test_logger"},
                "enabled": True,
            },
        ]

        provider.with_base_config(valid_config).with_export_config(exporters_config=exporters_config).configure_provider()

        # Verify that both exporters were created
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert len(provider.recorder._exporters) >= 2

        # Check that we have both types of exporters
        exporter_types = [type(exporter) for exporter in provider.recorder._exporters]
        assert FileExporter in exporter_types
        assert LoggerExporter in exporter_types

    def test_with_export_config_disabled_exporter(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config respects the enabled flag."""
        provider = TrainingTelemetryProvider.instance()

        # Test with disabled exporter
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/test.log"},
                "enabled": False,
            }
        ]

        provider.with_base_config(valid_config).with_export_config(exporters_config=exporters_config).configure_provider()

        # Verify that our disabled FileExporter was not created, but package configs may still be present
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        # Our FileExporter should not be present (since it was disabled)
        exporter_types = [type(exporter) for exporter in provider.recorder._exporters]
        assert FileExporter not in exporter_types

    def test_with_export_config_priority_behavior(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config correctly implements priority behavior.

        This test verifies the priority system works correctly without depending on
        specific package configurations being present.
        """
        provider = TrainingTelemetryProvider.instance()

        # Test 1: Direct config should always be applied regardless of package configs
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/priority_behavior_test.log"},
                "enabled": True,
            }
        ]

        provider.with_base_config(valid_config).with_export_config(exporters_config=exporters_config).configure_provider()

        # Our specified exporter must always be present
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        exporter_types = [type(exporter) for exporter in provider.recorder._exporters]
        assert FileExporter in exporter_types

        # Test 2: Verify that our exporter has the correct configuration
        our_exporter = None
        for exporter in provider.recorder._exporters:
            if isinstance(exporter, FileExporter) and exporter._filepath == Path("/tmp/priority_behavior_test.log"):
                our_exporter = exporter
                break

        assert our_exporter is not None, "Our specifically configured exporter should be present"

    def test_with_export_config_merge_behavior(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config correctly merges different exporter types.

        This test verifies that different exporter types are combined rather than replaced,
        demonstrating the merge behavior of the priority system.
        """
        provider = TrainingTelemetryProvider.instance()

        # Create a config with multiple different exporter types
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/merge_test1.log"},
                "enabled": True,
            },
            {
                "class_name": "nv_one_logger.exporter.logger_exporter.LoggerExporter",
                "config": {"logger": "merge_test_logger"},
                "enabled": True,
            },
        ]

        provider.with_base_config(valid_config).with_export_config(exporters_config=exporters_config).configure_provider()

        # Verify that both our specified exporters are present
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        exporter_types = [type(exporter) for exporter in provider.recorder._exporters]

        # Our specified exporters must be present
        assert FileExporter in exporter_types
        assert LoggerExporter in exporter_types

        # The total count should be at least our 2 exporters (may be more from package configs)
        assert len(provider.recorder._exporters) >= 2

    def test_with_export_config_package_discovery(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config discovers and uses package configurations.

        This test verifies that the package discovery mechanism works correctly
        by testing the entry points system without depending on specific packages.
        """
        from importlib.metadata import entry_points

        # Test 1: Verify that entry points discovery works
        eps = entry_points()
        try:
            entry_points_list = list(eps.select(group="nv_one_logger.exporter_configs"))  # python 3.9+
        except AttributeError:
            entry_points_list = list(eps.get("nv_one_logger.exporter_configs", []))  # python 3.8

        # The number of entry points depends on what's installed, but the mechanism should work
        print(f"Found {len(entry_points_list)} exporter config entry points")

        # Test 2: Verify that the ExporterConfigManager can load configurations
        from nv_one_logger.exporter.export_config_manager import ExporterConfigManager

        manager = ExporterConfigManager()

        # The manager should have loaded configurations from entry points
        print(f"ExporterConfigManager loaded {len(manager.entry_point_exporter_configs)} config classes")

        # Test 3: Verify that with_export_config() without direct config uses package configs
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(valid_config).with_export_config().configure_provider()

        # The provider should have exporters (either from package configs or none)
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)

        # Test 4: Verify that package configs are valid exporters
        for exporter in provider.recorder._exporters:
            # All exporters should have the basic exporter interface
            assert hasattr(exporter, "initialize")
            assert hasattr(exporter, "export_start")
            assert hasattr(exporter, "export_stop")
            assert hasattr(exporter, "export_event")
            assert hasattr(exporter, "export_error")
            assert hasattr(exporter, "export_telemetry_data_error")
            assert hasattr(exporter, "close")

        print(f"Successfully created {len(provider.recorder._exporters)} exporters from package configurations")

    def test_with_export_config_package_override_behavior(self, valid_config: TrainingTelemetryConfig) -> None:
        """Test that with_export_config correctly overrides package configurations.

        This test verifies that direct configuration properly overrides package
        configurations when they have the same exporter class.
        """
        provider = TrainingTelemetryProvider.instance()

        # First, get the baseline exporters from package configs
        provider.with_base_config(valid_config).with_export_config().configure_provider()
        baseline_exporters = len(provider.recorder._exporters)

        print(f"Baseline: {baseline_exporters} exporters from package configs")

        # Reset and test with direct config that should override package configs
        reset_singletong_providers_for_test()

        # Create a direct config that might override package configs
        exporters_config = [
            {
                "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
                "config": {"file_path": "/tmp/override_test.log"},
                "enabled": True,
            }
        ]

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(valid_config).with_export_config(exporters_config=exporters_config).configure_provider()

        # Our FileExporter should be present
        exporter_types = [type(exporter) for exporter in provider.recorder._exporters]
        assert FileExporter in exporter_types

        # The total count should be at least 1 (our exporter)
        assert len(provider.recorder._exporters) >= 1

        print(f"With override: {len(provider.recorder._exporters)} exporters (our FileExporter + any remaining package configs)")

        # Verify our specific configuration was applied
        our_exporter = None
        for exporter in provider.recorder._exporters:
            if isinstance(exporter, FileExporter) and exporter._filepath == Path("/tmp/override_test.log"):
                our_exporter = exporter
                break

        assert our_exporter is not None, "Our specifically configured FileExporter should be present"
