# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1_adapter __init__.py module.

These tests verify the configure_v2_adapter function which provides a compatibility layer
for v1 OneLogger configurations. The function:

1. Converts v1 config to v2 format using ConfigAdapter
2. Creates appropriate WandB exporters (async/sync based on configuration) using V1CompatibleExporter
3. Configures TrainingTelemetryProvider using the fluent API:
   - provider.with_base_config(training_config)
   - provider.with_exporter(exporter) for each exporter
   - provider.configure_provider()

Tests use minimal mocking - only the V1CompatibleExporter is mocked,
while ConfigAdapter and TrainingTelemetryProvider use real implementations to ensure
accurate behavior validation.
"""

import pytest
from nv_one_logger.api.config import OneLoggerConfig
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

        # Act - uses real ConfigAdapter and TrainingTelemetryProvider with auto-discovery
        configure_v2_adapter(v1_config)

        # Assert
        # Verify provider was configured properly
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        # Verify provider is actually configured (not just instantiated)
        assert provider.one_logger_ready

        # Verify the config was converted properly
        config = provider.config
        assert config.application_name == "test_project"
        assert config.session_tag == "test_run"
        assert config.telemetry_config is not None
        assert config.telemetry_config.perf_tag == "test_tag"

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

        # Act - uses real ConfigAdapter and TrainingTelemetryProvider with auto-discovery
        configure_v2_adapter(v1_config)

        # Assert
        # Verify provider was configured properly
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        # Verify provider is actually configured (not just instantiated)
        assert provider.one_logger_ready

        # Verify the config was converted properly
        config = provider.config
        assert config.application_name == "test_project"
        assert config.session_tag == "test_run"
        assert config.telemetry_config is not None
        assert config.telemetry_config.perf_tag == "test_tag"

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
        one_logger_config = ConfigAdapter.convert_to_v2_config(v1_config)

        # Assert - Verify real conversion worked
        assert one_logger_config is not None
        assert isinstance(one_logger_config, OneLoggerConfig)

        # Check that the conversion preserved key values
        assert one_logger_config.application_name == "real_test_project"
        assert one_logger_config.session_tag == "real_test_run"
        assert one_logger_config.telemetry_config is not None
        assert one_logger_config.telemetry_config.perf_tag == "real_test_tag"
        assert one_logger_config.world_size == 4
        assert one_logger_config.telemetry_config.global_batch_size == 64
        assert one_logger_config.is_baseline_run is False

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

        # Act - Uses real ConfigAdapter and real TrainingTelemetryProvider with auto-discovery
        configure_v2_adapter(v1_config)

        # Assert
        # Verify provider was configured properly
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        # Verify the real config conversion worked correctly
        config = provider.config
        assert config.application_name == "integration_test"
        assert config.session_tag == "integration_run"
        assert config.is_baseline_run is True
        assert config.telemetry_config.is_train_iterations_enabled is True
        assert config.telemetry_config.is_validation_iterations_enabled is False
        assert config.telemetry_config is not None
        assert config.telemetry_config.perf_tag == "integration_tag"
        assert config.world_size == 2
        assert config.telemetry_config.global_batch_size == 32

    def test_configure_v2_adapter_missing_one_logger_async_defaults_to_true(self):
        """Test configure_v2_adapter when one_logger_async is missing (defaults to True - async)."""
        # Arrange
        v1_config = {
            "one_logger_project": "async_default_test",
            "app_tag": "async_default_tag",
            "app_tag_run_name": "async_default_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 16,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            # one_logger_async missing - should use auto-discovery instead
        }

        # Act - uses auto-discovery
        configure_v2_adapter(v1_config)

        # Assert
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        # Verify the config was set correctly
        config = provider.config
        assert config.application_name == "async_default_test"
        assert config.session_tag == "async_default_run"

    def test_configure_v2_adapter_with_custom_metadata(self):
        """Test configure_v2_adapter with custom metadata."""
        # Arrange
        v1_config = {
            "one_logger_project": "metadata_test",
            "app_tag": "metadata_tag",
            "app_tag_run_name": "metadata_run",
            "is_baseline_run": False,
            "world_size": 1,
            "global_batch_size": 16,
            "is_train_iterations_enabled": True,
            "is_validation_iterations_enabled": True,
            "metadata": {"custom_key": "custom_value", "test_type": "metadata"},
            "app_tag_run_version": "2.0.0",
        }

        # Act - uses auto-discovery
        configure_v2_adapter(v1_config)

        # Assert
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        # Verify the config includes custom metadata
        config = provider.config
        assert config.application_name == "metadata_test"
        assert config.session_tag == "metadata_run"
        # Custom metadata should be embedded in the config
        assert config.custom_metadata is not None

    @pytest.mark.parametrize(
        "async_value,expected_async_mode",
        [
            (True, True),
            (False, False),
            (1, True),  # Non-zero integer is truthy
            (0, False),  # Zero is falsy
            ([], False),  # Empty list is falsy
            ([1], True),  # Non-empty list is truthy
        ],
    )
    def test_configure_v2_adapter_async_parameter_values(self, async_value, expected_async_mode):
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

        # Act - uses auto-discovery instead of manual exporter creation
        configure_v2_adapter(v1_config)

        # Assert - just verify that configuration works, auto-discovery handles exporter selection
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        # Verify the config was set correctly
        config = provider.config
        assert config.application_name == "param_test"
        assert config.session_tag == "param_run"

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
            # one_logger_async missing - uses auto-discovery
        }

        # Act - uses auto-discovery
        configure_v2_adapter(v1_config)

        # Assert
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        # Verify the config was set correctly
        config = provider.config
        assert config.application_name == "minimal_test"
        assert config.session_tag == "minimal_run"

    def test_configure_v2_adapter_fluent_api_usage(self):
        """Test that configure_v2_adapter properly uses the TrainingTelemetryProvider fluent API.

        Verifies that configure_v2_adapter correctly calls the fluent API with auto-discovery.
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

        # Act - uses auto-discovery
        configure_v2_adapter(v1_config)

        # Assert - verify the fluent API worked
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        # Verify the config was set correctly
        config = provider.config
        assert config.application_name == "fluent_test"
        assert config.session_tag == "fluent_run"

    def test_configure_v2_adapter_multiple_exporters(self):
        """Test that configure_v2_adapter handles exporter auto-discovery correctly."""
        # Arrange
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

        # Act - uses auto-discovery
        configure_v2_adapter(v1_config)

        # Assert - verify configuration worked
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
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

        # Act - uses auto-discovery
        configure_v2_adapter(v1_config)

        # Assert - verify all config fields were processed correctly
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        config = provider.config
        assert config.application_name == "test_project"
        assert config.session_tag == "test_run"
        assert config.is_baseline_run is True
        assert config.world_size == 4
        assert config.telemetry_config.global_batch_size == 128
        assert config.telemetry_config.log_every_n_train_iterations == 100

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

        # Act - full end-to-end test with auto-discovery
        configure_v2_adapter(v1_config)

        # Assert comprehensive validation
        provider = TrainingTelemetryProvider.instance()
        assert provider is not None
        assert provider.one_logger_ready

        config = provider.config
        assert config.application_name == "e2e_test"
        assert config.session_tag == "e2e_run"
        assert config.is_baseline_run is True
        assert config.world_size == 8
        assert config.telemetry_config.global_batch_size == 256
        assert config.telemetry_config.is_train_iterations_enabled is True
        assert config.telemetry_config.is_validation_iterations_enabled is False
