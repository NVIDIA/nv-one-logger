"""This module contains the compatibility layer for v2.

It allows switching from v1 to v2 of the one logger library without affecting the users of the library or the downstream consumers of
the telemetry data."""

from typing import Any, Dict

from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter
from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import (
    V1CompatibleWandbExporterAsync,
    V1CompatibleWandbExporterSync,
)


def configure_v2_adapter(v1_config: Dict[str, Any]) -> None:

    training_telemetry_config, wandb_config = ConfigAdapter.convert_to_v2_config(v1_config)

    enable_for_current_rank = v1_config.get("enable_for_current_rank", False)
    exporters: List[Exporter] = []
    if enable_for_current_rank:
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
    TrainingTelemetryProvider.instance().configure(config=training_telemetry_config, exporters=exporters)
