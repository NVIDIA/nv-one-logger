"""This module contains the compatibility layer for v1.

It allows switching from v1 to v2 of the one logger library without affecting the users of the library or the downstream consumers of
the telemetry data.

from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter
from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import (
    V1CompatibleExporter,
    V1CompatibleWandbExporterAdapter,
    V1CompatibleWandbExporterAsync,
    V1CompatibleWandbExporterSync,
)

__all__ = [
    "ConfigAdapter",
    "V1CompatibleExporter",
    "V1CompatibleWandbExporterAdapter",
    "V1CompatibleWandbExporterAsync",
    "V1CompatibleWandbExporterSync",
]
