"""This module contains the compatibility layer for v1.

It allows switching from v1 to v2 of the one logger library without affecting the users of the library or the downstream consumers of
the telemetry data.

"""

from typing import Any, Dict

from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter
from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import V1CompatibleExporter


def configure_v2_adapter(v1_config: Dict[str, Any]) -> None:
    """Configure the v2 adapter."""
    one_logger_config = ConfigAdapter.convert_to_v2_config(v1_config)

    # Configure the TrainingTelemetryProvider using the fluent API
    train_telemetry_provider_instance = TrainingTelemetryProvider.instance()

    # Set the base telemetry configuration and auto-discover exporters
    train_telemetry_provider_instance = train_telemetry_provider_instance.with_base_config(
        one_logger_config
    ).with_export_config()  # Auto-discover and configure exporters

    # Configure the provider to make it ready for use
    train_telemetry_provider_instance.configure_provider()


__all__ = [
    "ConfigAdapter",
    "V1CompatibleExporter",
    "configure_v2_adapter",
]
