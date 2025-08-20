# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExporterConfig:
    """Configuration for a single exporter.

    This dataclass represents the configuration for a single exporter instance.
    It contains the information needed to create and configure an exporter.

    Attributes:
        class_name: The full module path to the exporter class (e.g.,
            "nv_one_logger.exporter.file_exporter.FileExporter"). This is used
            for dynamic import and instantiation of the exporter.
        config: A dictionary containing configuration parameters for the exporter.
            The contents depend on the specific exporter type. For example,
            FileExporter might have {"file_path": "/tmp/log.txt"}.
        enabled: Whether this exporter should be created and used. If False,
            the exporter will be skipped during creation.
    """

    class_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
