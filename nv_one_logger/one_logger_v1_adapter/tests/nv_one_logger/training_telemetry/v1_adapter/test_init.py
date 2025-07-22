# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1_adapter __init__.py module.

These tests verify the configure_v2_adapter function which provides a compatibility layer
for v1 OneLogger configurations. The function:

1. Converts v1 config to v2 format using ConfigAdapter
2. Creates appropriate WandB exporters (async/sync based on configuration)
3. Configures TrainingTelemetryProvider using the fluent API:
   - provider.with_base_telemetry_config(training_config)
   - provider.with_exporter(exporter) for each exporter
   - provider.configure_provider()

Tests use minimal mocking - only external dependencies (WandB exporters) are mocked,
while ConfigAdapter and TrainingTelemetryProvider use real implementations to ensure
accurate behavior validation.
"""

from unittest.mock import Mock, patch

import pytest
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from nv_one_logger.training_telemetry.v1_adapter import configure_v2_adapter
from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset both TrainingTelemetryProvider and OneLoggerProvider singleton state before each test.

    This fixture resets all the internal state of both singletons:
    - Clears singleton instances
    - Resets configuration flags to False
    - Clears all temporary configuration state
    - Ensures clean state for each test
    """
    # Reset OneLoggerProvider singleton first (TrainingTelemetryProvider depends on it)
    OneLoggerProvider._instance = None
    one_logger_provider = OneLoggerProvider.instance()

    # Reset OneLoggerProvider configuration state - these are simple attributes, not name-mangled
    one_logger_provider._recorder = None  # type: ignore[reportPrivateUsage]
    one_logger_provider._config = None  # type: ignore[reportPrivateUsage]
    one_logger_provider._logging_force_disabled = False  # type: ignore[reportPrivateUsage]

    # Clear the TrainingTelemetryProvider singleton instance
    TrainingTelemetryProvider._instance = None

    # Get a fresh TrainingTelemetryProvider instance and reset its configuration state
    training_provider = TrainingTelemetryProvider.instance()

    # Reset the main configuration flag (name-mangled private attribute)
    training_provider._TrainingTelemetryProvider__fully_configured = False  # type: ignore[reportPrivateUsage]

    # Reset temporary configuration state
    temp_attrs = [
        "_TrainingTelemetryProvider__tmp_base_config",
        "_TrainingTelemetryProvider__tmp_config_overrides",
        "_TrainingTelemetryProvider__tmp_exporters",
        "_TrainingTelemetryProvider__tmp_export_customization_mode",
        "_TrainingTelemetryProvider__tmp_span_name_filter",
    ]

    for attr_name in temp_attrs:
        if hasattr(training_provider, attr_name):
            current_value = getattr(training_provider, attr_name)
            if attr_name.endswith("__tmp_base_config") or attr_name.endswith("__tmp_export_customization_mode") or attr_name.endswith("__tmp_span_name_filter"):
                setattr(training_provider, attr_name, None)
            elif attr_name.endswith("__tmp_config_overrides"):
                if isinstance(current_value, dict):
                    current_value.clear()
            elif attr_name.endswith("__tmp_exporters"):
                if isinstance(current_value, list):
                    current_value.clear()

    # Debug: uncomment if you need to verify singleton reset behavior
    # print(f"Singletons reset - OneLogger and Training providers cleaned")

    yield  # This is where the test runs

    # Cleanup after test
    TrainingTelemetryProvider._instance = None
    OneLoggerProvider._instance = None


class TestConfigureV2Adapter:
    """Test cases for configure_v2_adapter function."""

    def test_configure_v2_adapter_with_async_exporter(self):
        """Test configure_v2_adapter with async exporter using real components."""
        # Arrange
        v1_config = {
            "one_logger_async": True,
            "one_logger_project": "test_project",
            "app_tag": "test_tag",
            "app_tag_run_name": "test_run",
            "is_baseline_run": False,
            "world_size": 2,
            "global_batch_size": 32,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterAsync") as mock_async_exporter_class:
            # Setup mocks - only mock external dependencies
            mock_async_exporter = Mock()
            mock_async_exporter_class.return_value = mock_async_exporter

            # Act - uses real ConfigAdapter and TrainingTelemetryProvider
            configure_v2_adapter(v1_config)

            # Assert
            # Verify the real config adapter was used
            mock_async_exporter_class.assert_called_once()
            call_args = mock_async_exporter_class.call_args

            # Verify the exporter was created with proper config objects
            assert call_args.kwargs["training_telemetry_config"] is not None
            assert call_args.kwargs["wandb_config"] is not None

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_with_sync_exporter(self):
        """Test configure_v2_adapter with sync exporter using real components."""
        # Arrange
        v1_config = {
            "one_logger_async": False,
            "one_logger_project": "test_project",
            "app_tag": "test_tag",
            "app_tag_run_name": "test_run",
            "is_baseline_run": False,
            "world_size": 2,
            "global_batch_size": 32,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class:
            # Setup mocks - only mock external dependencies
            mock_sync_exporter = Mock()
            mock_sync_exporter_class.return_value = mock_sync_exporter

            # Act - uses real ConfigAdapter and TrainingTelemetryProvider
            configure_v2_adapter(v1_config)

            # Assert
            # Verify the real config adapter was used
            mock_sync_exporter_class.assert_called_once()
            call_args = mock_sync_exporter_class.call_args

            # Verify the exporter was created with proper config objects
            assert call_args.kwargs["training_telemetry_config"] is not None
            assert call_args.kwargs["wandb_config"] is not None

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_real_config_conversion(self):
        """Test that real ConfigAdapter.convert_to_v2_config works correctly with actual config."""
        # Arrange
        v1_config = {
            "one_logger_project": "real_test_project",
            "app_tag": "real_test_tag",
            "app_tag_run_name": "real_test_run",
            "is_baseline_run": False,
            "world_size": 4,
            "global_batch_size": 64,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "one_logger_async": True,
        }

        # Act - Use real ConfigAdapter directly
        training_config, wandb_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Assert - Verify real conversion worked
        assert training_config is not None
        assert wandb_config is not None

        # Check that the conversion preserved key values
        assert training_config.application_name == "real_test_project"
        assert training_config.perf_tag_or_fn == "real_test_tag"
        assert training_config.session_tag_or_fn == "real_test_run"
        assert training_config.world_size_or_fn == 4
        assert training_config.global_batch_size_or_fn == 64
        assert training_config.is_baseline_run_or_fn is False

    def test_configure_v2_adapter_integration_with_real_configs(self):
        """Integration test using real ConfigAdapter with minimal mocking."""
        # Arrange
        v1_config = {
            "one_logger_project": "integration_test",
            "app_tag": "integration_tag",
            "app_tag_run_name": "integration_run",
            "is_baseline_run": True,
            "world_size": 2,
            "global_batch_size": 32,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": False,
            "one_logger_async": False,  # Test sync path
        }

        # Only mock exporter classes (external dependencies)
        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class:
            mock_sync_exporter = Mock()
            mock_sync_exporter_class.return_value = mock_sync_exporter

            # Act - Uses real ConfigAdapter and real TrainingTelemetryProvider
            configure_v2_adapter(v1_config)

            # Assert
            # Verify exporter was created with real converted configs
            mock_sync_exporter_class.assert_called_once()
            call_args = mock_sync_exporter_class.call_args.kwargs

            # The configs should be real objects from ConfigAdapter
            training_config = call_args["training_telemetry_config"]
            wandb_config = call_args["wandb_config"]

            assert training_config.application_name == "integration_test"
            assert training_config.perf_tag_or_fn == "integration_tag"
            assert training_config.session_tag_or_fn == "integration_run"
            assert training_config.is_baseline_run_or_fn is True
            assert training_config.world_size_or_fn == 2
            assert training_config.global_batch_size_or_fn == 32
            assert training_config.is_train_iterations_enabled_or_fn is True
            assert training_config.is_validation_iterations_enabled_or_fn is False
            assert wandb_config is not None

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_missing_one_logger_async_defaults_to_true(self):
        """Test configure_v2_adapter when one_logger_async is missing (defaults to True - async)."""
        # Arrange
        v1_config = {
            # Missing "one_logger_async" key - should default to True
            "one_logger_project": "test_project",
            "app_tag": "test_tag",
            "app_tag_run_name": "test_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 16,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
        }

        with (
            patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterAsync") as mock_async_exporter_class,
            patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class,
        ):
            # Setup mocks
            mock_async_exporter = Mock()
            mock_async_exporter_class.return_value = mock_async_exporter

            # Act - uses real TrainingTelemetryProvider and ConfigAdapter
            configure_v2_adapter(v1_config)

            # Assert
            # Should use async exporter (default)
            mock_async_exporter_class.assert_called_once()
            mock_sync_exporter_class.assert_not_called()

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_with_custom_metadata(self):
        """Test that custom metadata and additional config fields are properly handled by real ConfigAdapter."""
        # Arrange
        v1_config = {
            "one_logger_async": False,
            "one_logger_project": "test_project",
            "app_tag": "test_tag",
            "app_tag_run_name": "test_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 8,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "metadata": {"custom_key": "custom_value"},
            "app_tag_run_version": "2.0.0",
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class:
            # Setup mocks
            mock_sync_exporter = Mock()
            mock_sync_exporter_class.return_value = mock_sync_exporter

            # Act - uses real ConfigAdapter and TrainingTelemetryProvider
            configure_v2_adapter(v1_config)

            # Assert - verify real config conversion worked with custom metadata
            mock_sync_exporter_class.assert_called_once()
            call_args = mock_sync_exporter_class.call_args.kwargs
            training_config = call_args["training_telemetry_config"]

            # Verify real ConfigAdapter processed custom metadata
            assert "custom_key" in training_config.custom_metadata
            assert training_config.custom_metadata["custom_key"] == "custom_value"
            assert training_config.custom_metadata["app_tag_run_version"] == "2.0.0"

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    @pytest.mark.parametrize(
        "async_value,expected_exporter_class",
        [
            (True, "V1CompatibleWandbExporterAsync"),
            (False, "V1CompatibleWandbExporterSync"),
            (1, "V1CompatibleWandbExporterAsync"),  # Non-zero integer is truthy
            (0, "V1CompatibleWandbExporterSync"),  # Zero is falsy
            ([], "V1CompatibleWandbExporterSync"),  # Empty list is falsy
            ([1], "V1CompatibleWandbExporterAsync"),  # Non-empty list is truthy
        ],
    )
    def test_configure_v2_adapter_async_parameter_values(self, async_value, expected_exporter_class):
        """Test configure_v2_adapter with various values for one_logger_async parameter."""
        # Arrange
        v1_config = {
            "one_logger_async": async_value,
            "one_logger_project": "param_test",
            "app_tag": "param_tag",
            "app_tag_run_name": "param_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 16,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
        }

        with (
            patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterAsync") as mock_async_exporter_class,
            patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class,
        ):
            # Setup mocks
            mock_exporter = Mock()
            mock_async_exporter_class.return_value = mock_exporter
            mock_sync_exporter_class.return_value = mock_exporter

            # Act - uses real TrainingTelemetryProvider and ConfigAdapter
            configure_v2_adapter(v1_config)

            # Assert
            if expected_exporter_class == "V1CompatibleWandbExporterAsync":
                mock_async_exporter_class.assert_called_once()
                mock_sync_exporter_class.assert_not_called()
            else:
                mock_sync_exporter_class.assert_called_once()
                mock_async_exporter_class.assert_not_called()

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_exception_handling_missing_config_fields(self):
        """Test configure_v2_adapter behavior when ConfigAdapter raises exceptions for missing fields."""
        # Arrange - Use config missing required fields to trigger real exception from ConfigAdapter
        v1_config = {"invalid_config": "missing required fields"}

        # Act & Assert - Real ConfigAdapter should raise KeyError for missing fields
        with pytest.raises(KeyError):
            configure_v2_adapter(v1_config)

    def test_configure_v2_adapter_with_minimal_config(self):
        """Test configure_v2_adapter with minimal required config."""
        # Arrange
        v1_config = {
            # Only required fields for ConfigAdapter
            "one_logger_project": "minimal_test",
            "app_tag": "minimal_tag",
            "app_tag_run_name": "minimal_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 8,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            # one_logger_async missing - should default to True (async)
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterAsync") as mock_async_exporter_class:
            # Setup mocks
            mock_async_exporter = Mock()
            mock_async_exporter_class.return_value = mock_async_exporter

            # Act
            configure_v2_adapter(v1_config)

            # Assert
            # Should default to async exporter (one_logger_async defaults to True)
            mock_async_exporter_class.assert_called_once()

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_fluent_api_usage(self):
        """Test that configure_v2_adapter properly uses the TrainingTelemetryProvider fluent API.

        Verifies that configure_v2_adapter correctly calls:
        1. with_base_telemetry_config() with the converted training config
        2. with_exporter() for each exporter
        3. configure_provider() to finalize configuration
        """
        # Arrange
        v1_config = {
            "one_logger_async": False,
            "one_logger_project": "fluent_test",
            "app_tag": "fluent_tag",
            "app_tag_run_name": "fluent_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 8,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class:
            mock_sync_exporter = Mock()
            mock_sync_exporter_class.return_value = mock_sync_exporter

            # Spy on the provider methods to verify fluent API usage
            provider = TrainingTelemetryProvider.instance()

            with (
                patch.object(provider, "with_base_telemetry_config", wraps=provider.with_base_telemetry_config) as spy_with_config,
                patch.object(provider, "with_exporter", wraps=provider.with_exporter) as spy_with_exporter,
                patch.object(provider, "configure_provider", wraps=provider.configure_provider) as spy_configure,
            ):
                # Act
                configure_v2_adapter(v1_config)

                # Assert - verify the fluent API was used correctly

                # Should call with_base_telemetry_config with the converted training config
                spy_with_config.assert_called_once()
                call_args = spy_with_config.call_args[0][0]  # Get first positional argument
                assert call_args is not None
                assert call_args.application_name == "fluent_test"

                # Should call with_exporter
                spy_with_exporter.assert_called_once_with(mock_sync_exporter)

                # Should call configure_provider
                spy_configure.assert_called_once()

                # Verify final state
                assert provider.one_logger_ready

    def test_configure_v2_adapter_multiple_exporters(self):
        """Test that configure_v2_adapter handles multiple exporters correctly.

        This test verifies that when multiple exporters are configured,
        the fluent API correctly chains all with_exporter() calls.
        """
        # Arrange - This config will result in one exporter, but we'll simulate multiple
        v1_config = {
            "one_logger_async": False,
            "one_logger_project": "multi_test",
            "app_tag": "multi_tag",
            "app_tag_run_name": "multi_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 8,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class:
            mock_sync_exporter = Mock()
            mock_sync_exporter_class.return_value = mock_sync_exporter

            # Spy on the provider methods to verify multiple exporter handling
            provider = TrainingTelemetryProvider.instance()

            with patch.object(provider, "with_exporter", wraps=provider.with_exporter) as spy_with_exporter:
                # Act
                configure_v2_adapter(v1_config)

                # Assert
                # Should call with_exporter once per exporter (in this case, 1)
                spy_with_exporter.assert_called_once_with(mock_sync_exporter)

                # Verify provider is properly configured
                assert provider.one_logger_ready

    def test_configure_v2_adapter_with_additional_config_fields(self):
        """Test configure_v2_adapter with additional optional config fields."""
        # Arrange
        v1_config = {
            "one_logger_async": False,
            "one_logger_project": "test_project",
            "app_tag": "test_tag",
            "app_tag_run_name": "test_run",
            "is_baseline_run": True,
            "world_size": 4,
            "global_batch_size": 128,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": False,
            # Additional optional fields
            "quiet": True,
            "log_every_n_train_iterations": 100,
            "is_test_iterations_enabled": False,
            "is_save_checkpoint_enabled": True,
            "is_log_throughput_enabled": True,
            "micro_batch_size": 8,
            "flops_per_sample": 1000,
            "save_checkpoint_strategy": "sync",
            "train_iterations_target": 10000,
            "train_samples_target": 100000,
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterSync") as mock_sync_exporter_class:
            # Setup mocks
            mock_sync_exporter = Mock()
            mock_sync_exporter_class.return_value = mock_sync_exporter

            # Act
            configure_v2_adapter(v1_config)

            # Assert
            mock_sync_exporter_class.assert_called_once()
            call_args = mock_sync_exporter_class.call_args.kwargs
            training_config = call_args["training_telemetry_config"]

            # Verify additional fields were processed correctly by real ConfigAdapter
            assert training_config.log_every_n_train_iterations == 100
            assert training_config.is_test_iterations_enabled_or_fn is False
            assert training_config.is_save_checkpoint_enabled_or_fn is True
            assert training_config.is_log_throughput_enabled_or_fn is True
            assert training_config.micro_batch_size_or_fn == 8
            assert training_config.flops_per_sample_or_fn == 1000
            assert training_config.train_iterations_target_or_fn == 10000
            assert training_config.train_samples_target_or_fn == 100000

            # Verify real provider was configured
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            # Verify provider is actually configured (not just instantiated)
            assert provider.one_logger_ready

    def test_configure_v2_adapter_end_to_end_validation(self):
        """Comprehensive end-to-end test validating the complete configure_v2_adapter flow."""
        # Arrange
        v1_config = {
            "one_logger_async": True,
            "one_logger_project": "e2e_test",
            "app_tag": "e2e_tag",
            "app_tag_run_name": "e2e_run",
            "is_baseline_run": True,
            "world_size": 8,
            "global_batch_size": 256,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": False,
            "metadata": {"test_type": "end_to_end"},
            "app_tag_run_version": "1.0.0",
            "quiet": False,
        }

        with patch("nv_one_logger.training_telemetry.v1_adapter.V1CompatibleWandbExporterAsync") as mock_async_exporter_class:
            mock_async_exporter = Mock()
            mock_async_exporter_class.return_value = mock_async_exporter

            # Act - Full end-to-end execution
            configure_v2_adapter(v1_config)

            # Assert - Comprehensive validation

            # 1. Verify exporter was created with real converted configs
            mock_async_exporter_class.assert_called_once()
            call_args = mock_async_exporter_class.call_args.kwargs
            training_config = call_args["training_telemetry_config"]
            wandb_config = call_args["wandb_config"]

            # 2. Verify ConfigAdapter conversion worked correctly
            assert training_config.application_name == "e2e_test"
            assert training_config.perf_tag_or_fn == "e2e_tag"
            assert training_config.session_tag_or_fn == "e2e_run"
            assert training_config.is_baseline_run_or_fn is True
            assert training_config.world_size_or_fn == 8
            assert training_config.global_batch_size_or_fn == 256
            assert training_config.is_train_iterations_enabled_or_fn is True
            assert training_config.is_validation_iterations_enabled_or_fn is False
            assert "test_type" in training_config.custom_metadata
            assert training_config.custom_metadata["test_type"] == "end_to_end"
            assert training_config.custom_metadata["app_tag_run_version"] == "1.0.0"
            assert wandb_config is not None

            # 3. Verify provider is fully configured and ready
            provider = TrainingTelemetryProvider.instance()
            assert provider is not None
            assert provider.one_logger_ready

            # 4. Verify provider has access to the configuration
            provider_config = provider.config
            assert provider_config.application_name == "e2e_test"
            assert provider_config.world_size_or_fn == 8

            # 5. Verify provider has access to the recorder
            recorder = provider.recorder
            assert recorder is not None
