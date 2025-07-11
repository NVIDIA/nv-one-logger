# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
"""Unit tests for the TrainingTelemetryProvider class.

This module contains tests that verify the functionality of the TrainingTelemetryProvider class,
including configuration, recorder management, and singleton behavior.
"""

from typing import Generator, List, cast
from unittest.mock import MagicMock

import pytest
from nv_one_logger.api.config import ApplicationType
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.span import SpanName
from nv_one_logger.exporter.exporter import Exporter
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


_BASE_CONFIG = TrainingTelemetryConfig(
    enable_for_current_rank=True,
    world_size_or_fn=4,
    global_batch_size_or_fn=32,
    application_name="test_app",
    perf_tag_or_fn="test_perf",
    session_tag_or_fn="test_session",
    app_type_or_fn=ApplicationType.TRAINING,
    is_baseline_run_or_fn=False,
    save_checkpoint_strategy=CheckPointStrategy.SYNC,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Reset the singleton instance of TrainingTelemetryProvider before each test."""
    reset_singletong_providers_for_test()
    yield
    # Reset the singleton after the test
    reset_singletong_providers_for_test()


@pytest.fixture
def valid_config() -> TrainingTelemetryConfig:
    """Fixture that returns a valid TrainingTelemetryConfig."""
    return TrainingTelemetryConfig(
        world_size_or_fn=4,
        global_batch_size_or_fn=32,
        application_name="test_app",
        perf_tag_or_fn="test_perf",
        session_tag_or_fn="test_session",
        app_type_or_fn=ApplicationType.TRAINING,
        is_baseline_run_or_fn=False,
        save_checkpoint_strategy=CheckPointStrategy.SYNC,
        enable_for_current_rank=True,
    )


class TestTrainingTelemetryProvider:
    """Tests for TrainingTelemetryProvider class."""

    def test_singleton_behavior(self) -> None:
        """Test that TrainingTelemetryProvider behaves as a singleton."""
        provider1 = TrainingTelemetryProvider.instance()
        provider2 = TrainingTelemetryProvider.instance()
        assert provider1 is provider2

    def test_configure_with_disabled_telemetry(self, valid_config: TrainingTelemetryConfig, mock_exporter: Exporter) -> None:
        """Test configuration when telemetry is disabled for current rank."""
        disabled_config = valid_config.model_copy()
        disabled_config.enable_for_current_rank = False

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(disabled_config).with_exporter(mock_exporter).configure_provider()
        assert provider.config == disabled_config
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._exporters == []  # Force the exporter to be empty

    def test_with_base_telemetry_config_success(self, mock_exporter: Exporter) -> None:
        """Test that with_base_telemetry_config sets the base config correctly."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(_BASE_CONFIG).with_exporter(mock_exporter).configure_provider()
        assert provider.config == _BASE_CONFIG
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._exporters == [mock_exporter]

    def test_with_base_telemetry_config_called_twice_raises_error(self) -> None:
        """Test that calling with_base_telemetry_config twice raises an error."""
        another_config = TrainingTelemetryConfig(
            world_size_or_fn=80,
            global_batch_size_or_fn=400,
            application_name="test_app2",
            perf_tag_or_fn="test_perf2",
            session_tag_or_fn="test_session2",
            app_type_or_fn=ApplicationType.TRAINING,
            is_baseline_run_or_fn=False,
            save_checkpoint_strategy=CheckPointStrategy.SYNC,
        )
        with pytest.raises(OneLoggerError, match="You can only call with_base_telemetry_config once"):
            TrainingTelemetryProvider.instance().with_base_telemetry_config(_BASE_CONFIG).with_base_telemetry_config(another_config)

    def test_build_telemetry_config_with_base_config_override(self) -> None:
        """Test that _build_telemetry_config works correctly with a base config."""
        TrainingTelemetryProvider.instance().with_base_telemetry_config(_BASE_CONFIG).with_config_override(
            {
                "world_size_or_fn": 8,
                "log_every_n_train_iterations": 100,
            }
        ).configure_provider()
        result_config = TrainingTelemetryProvider.instance().config
        assert isinstance(result_config, TrainingTelemetryConfig)
        assert result_config.world_size == 8  # Overridden value
        assert result_config.log_every_n_train_iterations == 100  # Overridden value
        assert result_config.global_batch_size == 32  # base value
        assert result_config.application_name == "test_app"  # base value

    def test_with_incomplete_config_raises_error(self) -> None:
        """Test that if we don't provide required fields, the builder raises an error."""
        override1 = {"log_every_n_train_iterations": 100}
        override2 = {"enable_for_current_rank": True}
        with pytest.raises(OneLoggerError, match="Invalid configuration!"):
            TrainingTelemetryProvider.instance().with_config_override(override1).with_config_override(override2).configure_provider()

    def test_with_config_override_updates_existing_keys(self) -> None:
        """Test that with_config_override updates existing keys correctly."""
        override1 = {"log_every_n_train_iterations": 100}
        override2 = {"log_every_n_train_iterations": 200}  # Override the same key
        TrainingTelemetryProvider.instance().with_base_telemetry_config(_BASE_CONFIG).with_config_override(override1).with_config_override(
            override2
        ).configure_provider()
        assert TrainingTelemetryProvider.instance().config.log_every_n_train_iterations == 200

    def test_with_multiple_exporter_success(self, mock_exporter: Exporter, another_mock_exporter: Exporter) -> None:
        """Test that with_exporter adds exporters correctly."""
        TrainingTelemetryProvider.instance().with_base_telemetry_config(_BASE_CONFIG).with_exporter(mock_exporter).with_exporter(
            another_mock_exporter
        ).configure_provider()
        assert TrainingTelemetryProvider.instance().recorder._exporters == [mock_exporter, another_mock_exporter]

    def test_no_exporters_success(self) -> None:
        """Test that if we don't provide any exporters, the builder doesn't raise an error."""
        TrainingTelemetryProvider.instance().with_base_telemetry_config(_BASE_CONFIG).configure_provider()
        provider = TrainingTelemetryProvider.instance()
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._exporters == []

    def test_no_config_raises_error(self, mock_exporter: Exporter) -> None:
        """Test that if we don't provide any config, the builder raises an error."""
        with pytest.raises(OneLoggerError, match="No configuration was provided. Please provide a base config and/or config overrides."):
            TrainingTelemetryProvider.instance().with_exporter(mock_exporter).configure_provider()

    def test_build_telemetry_config_without_base_config(self) -> None:
        """Test that _build_telemetry_config works correctly without a base config if enough config overrides are provided."""
        override = {
            "world_size_or_fn": 8,
            "global_batch_size_or_fn": 64,
            "application_name": "test_app",
            "perf_tag_or_fn": "test_perf",
            "session_tag_or_fn": "test_session",
            "app_type_or_fn": ApplicationType.TRAINING,
            "is_baseline_run_or_fn": False,
            "save_checkpoint_strategy": CheckPointStrategy.SYNC,
        }

        TrainingTelemetryProvider.instance().with_config_override(override).configure_provider()
        result_config = TrainingTelemetryProvider.instance().config
        assert isinstance(result_config, TrainingTelemetryConfig)
        assert result_config.world_size == 8
        assert result_config.global_batch_size == 64
        assert result_config.application_name == "test_app"

    def test_build_telemetry_config_with_multiple_exporters(self, mock_exporter: Exporter, another_mock_exporter: Exporter) -> None:
        """Test that _build_telemetry_config works correctly with multiple exporters."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(_BASE_CONFIG).with_exporter(mock_exporter).with_exporter(another_mock_exporter).configure_provider()
        assert provider.recorder._exporters == [mock_exporter, another_mock_exporter]

    def test_build_telemetry_config_invalid_config_raises_error(self) -> None:
        """Test that _build_telemetry_config raises an error for invalid configuration."""
        invalid_override = {
            "world_size_or_fn": 0,  # Invalid: must be > 0
            "global_batch_size_or_fn": 32,
            "application_name": "test_app",
            "perf_tag_or_fn": "test_perf",
            "session_tag_or_fn": "test_session",
            "app_type_or_fn": ApplicationType.TRAINING,
            "is_baseline_run_or_fn": False,
            "save_checkpoint_strategy": CheckPointStrategy.SYNC,
        }

        with pytest.raises(OneLoggerError, match="world_size must be set to a non-zero value"):
            TrainingTelemetryProvider.instance().with_base_telemetry_config(_BASE_CONFIG).with_config_override(invalid_override).configure_provider()

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
            provider.with_base_telemetry_config(_BASE_CONFIG)
            .with_exporter(mock_exporter)
            .with_exporter(another_mock_exporter)
            .with_export_customization(export_customization_mode=ExportCustomizationMode.WHITELIST_SPANS, span_name_filter=custom_span_filter)
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
                .with_base_telemetry_config(_BASE_CONFIG)
                .with_export_customization(export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter=custom_span_filter)
                .with_export_customization(export_customization_mode=ExportCustomizationMode.WHITELIST_SPANS, span_name_filter=another_span_filter)
            )

    def test_with_config_override_after_configure_provider_raises_error(self) -> None:
        """Test that calling with_config_override after configure_provider raises an error."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(_BASE_CONFIG).configure_provider()
        with pytest.raises(OneLoggerError, match="with_config_override can be called only before configure_provider is called."):
            provider.with_config_override({"log_every_n_train_iterations": 100})

    def test_with_exporter_after_configure_provider_raises_error(self, mock_exporter: Exporter) -> None:
        """Test that calling with_exporter after configure_provider raises an error."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(_BASE_CONFIG).configure_provider()
        with pytest.raises(OneLoggerError, match="with_exporter can be called only before configure_provider is called."):
            provider.with_exporter(mock_exporter)

    def test_with_export_customization_after_configure_raises_error(self) -> None:
        """Test that calling with_export_customization after configure_provider raises an error."""
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(_BASE_CONFIG).configure_provider()

        custom_span_filter = cast(List[SpanName], [StandardTrainingJobSpanName.TRAINING_LOOP])
        with pytest.raises(OneLoggerError, match="with_export_customization can be called only before configure_provider is called."):
            provider.with_export_customization(export_customization_mode=ExportCustomizationMode.BLACKLIST_SPANS, span_name_filter=custom_span_filter)

    def test_configure_provider_without_export_customization_uses_defaults(self, mock_exporter: Exporter) -> None:
        """Test that configure_provider uses default export customization when not specified."""
        from nv_one_logger.training_telemetry.api.training_telemetry_provider import DEFAULT_SPANS_EXPORT_BLACKLIST

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_telemetry_config(_BASE_CONFIG).with_exporter(mock_exporter).configure_provider()

        # Verify the recorder was created with the default export customization settings
        assert provider.recorder and isinstance(provider.recorder, TrainingRecorder)
        assert provider.recorder._span_name_filter == DEFAULT_SPANS_EXPORT_BLACKLIST
        assert provider.recorder._export_customization_mode == ExportCustomizationMode.BLACKLIST_SPANS
