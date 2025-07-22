"""This module contains the compatibility layer for v1.

It allows switching from v1 to v2 of the one logger library without affecting the users of the library or the downstream consumers of
the telemetry data.

"""

from typing import Any, Dict, List

from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter
from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import (
    V1CompatibleExporter,
    V1CompatibleWandbExporterAdapter,
    V1CompatibleWandbExporterAsync,
    V1CompatibleWandbExporterSync,
)


def configure_v2_adapter(v1_config: Dict[str, Any]) -> None:
    """Configure the v2 adapter."""
    training_telemetry_config, wandb_config = ConfigAdapter.convert_to_v2_config(v1_config)

    exporters: List[Exporter] = []
    if v1_config.get("one_logger_async", True):
        exporters.append(
            V1CompatibleWandbExporterAsync(
                training_telemetry_config=training_telemetry_config,
                wandb_config=wandb_config,
            )
        )
    else:
        exporters.append(
            V1CompatibleWandbExporterSync(
                training_telemetry_config=training_telemetry_config,
                wandb_config=wandb_config,
            )
        )
    # Configure the TrainingTelemetryProvider using the fluent API
    train_telemetry_provider_instance = TrainingTelemetryProvider.instance()

    # Set the base telemetry configuration (only once)
    train_telemetry_provider_instance = train_telemetry_provider_instance.with_base_telemetry_config(training_telemetry_config)

    # Add all exporters
    for exporter in exporters:
        train_telemetry_provider_instance = train_telemetry_provider_instance.with_exporter(exporter)

    # Configure the provider to make it ready for use
    train_telemetry_provider_instance.configure_provider()


__all__ = [
    "ConfigAdapter",
    "V1CompatibleExporter",
    "V1CompatibleWandbExporterAdapter",
    "V1CompatibleWandbExporterAsync",
    "V1CompatibleWandbExporterSync",
    "configure_v2_adapter",
]
