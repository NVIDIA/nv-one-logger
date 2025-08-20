# SPDX-License-Identifier: Apache-2.0
import json
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.exporter.exporter_config import ExporterConfig

_logger = get_logger(__name__)


class ExporterConfigManager:
    """Manages exporter configuration with priority-based loading.

    This class handles the discovery, loading, and merging of exporter configurations
    from multiple sources with a defined priority order:
    1. Direct configuration (highest priority)
    2. File configuration (medium priority)
    3. Package configuration (lowest priority)

    The class automatically discovers exporter configurations from installed packages
    using Python entry points and provides methods to build final configurations
    and create exporter instances.
    """

    # Entry point name for discovering exporter configurations
    ENTRY_POINT_NAME = "nv_one_logger.exporter_configs"

    def __init__(self):
        """Initialize the ExporterConfigManager.

        Automatically loads exporter configurations from installed packages
        using the entry point system. Any packages that register themselves
        under the ENTRY_POINT_NAME will have their configurations loaded
        and available for use.

        Raises:
                None: Exceptions during entry point loading are logged as warnings
                      and do not prevent initialization.
        """
        self.entry_point_exporter_configs = self._get_exporter_configs_from_entry_points()

    def generate_export_config(
        self,
        direct_config: Optional[Union[List[ExporterConfig], List[Dict[str, Any]]]] = None,
        config_file_path: Optional[str] = None,
    ) -> List[ExporterConfig]:
        """Generate export configuration using priority order.

        This method merges configurations from multiple sources according to the
        defined priority system. Higher priority configurations override lower
        priority ones, while different exporter types are combined.

        Args:
                direct_config: Configuration passed directly to the method. Can be:
                        - List[ExporterConfig]: A list of ExporterConfig objects
                        - List[Dict[str, Any]]: A list of dictionaries, each representing an exporter config

                        If a list of dictionaries is provided, it will be automatically
                        converted to a list of ExporterConfig objects. This has the highest priority
                        and will override any conflicting configurations from other sources.
                config_file_path: Path to a configuration file (YAML or JSON). If
                        provided, this file will be loaded and merged with other
                        configurations. If None, the system will look for configuration
                        files in the current working directory or use the environment
                        variable ONE_LOGGER_EXPORTER_CONFIG_PATH.

        Returns:
                List[ExporterConfig]: The merged configuration containing all
                exporters from all sources, with higher priority configurations
                taking precedence over lower priority ones.

        Note:
                The priority order is:
                1. Direct configuration (highest priority)
                2. File configuration (medium priority)
                3. Package configuration (lowest priority)

                For the same exporter class, higher priority configurations completely
                replace lower priority ones. For different exporter classes, all
                configurations are combined.
        """
        # Start with package/team configs (lowest priority)
        base_config = self._get_package_configs()
        if base_config:
            _logger.info(f"Loaded {len(base_config)} exporter config(s) from packages")

        # Override with file config (medium priority)
        file_config = self._get_file_config(config_file_path)
        if file_config:
            _logger.info(f"Loaded {len(file_config)} exporter config(s) from file")
            # File config completely replaces base config, rather than merging
            base_config = file_config

        # Override with direct config (highest priority)
        if direct_config:
            # Convert to list of ExporterConfig if needed
            if isinstance(direct_config, list) and direct_config and isinstance(direct_config[0], dict):
                # Handle list of dictionaries
                direct_config = self._build_exporter_configs(direct_config)
            base_config = self._merge_configs(base_config, direct_config)

        _logger.info(f"Final configuration contains {len(base_config)} exporter(s)")
        return base_config

    def create_exporters_from_config(self, exporters_config: List[ExporterConfig], training_telemetry_config: Optional[Any] = None) -> List[Exporter]:
        """Create exporter instances from configuration.

        This method takes a list of ExporterConfig objects and creates actual exporter
        instances from the configuration. Only enabled exporters are created,
        and any failures during creation are logged as warnings.

        Args:
                exporters_config: The list of ExporterConfig objects containing exporter
                        definitions. Each exporter in the configuration must have a
                        valid class_name that can be dynamically imported.
                training_telemetry_config: Optional training telemetry configuration that
                        will be passed to exporters that require it (e.g., V1CompatibleExporter).

        Returns:
                List[Exporter]: A list of created exporter instances. The list may
                        be empty if no exporters are enabled or if all exporters fail
                        to create. Each exporter in the list is a valid Exporter instance
                        that can be used for telemetry data export.

        Note:
                - Only exporters with enabled=True are created
                - Failed exporter creation is logged as warnings and skipped
                - The method uses dynamic import to load exporter classes
                - Special handling is provided for FileExporter to convert
                  string paths to Path objects
        """
        exporters = []

        # Create exporters from config only (no default nv_exporter)
        for exporter_config in exporters_config:
            if not exporter_config.enabled:
                continue

            exporter = self._create_exporter_from_config(exporter_config, training_telemetry_config)
            if exporter:
                exporters.append(exporter)

        return exporters

    # =============================================================================
    # PRIVATE IMPLEMENTATION METHODS
    # =============================================================================

    def _get_exporter_configs_from_entry_points(self) -> Dict[str, Any]:
        """Load exporter configurations from entry points (package/team configs)."""
        configs = {}
        try:
            for entry_point in entry_points(group=self.ENTRY_POINT_NAME):
                try:
                    config_class = entry_point.load()
                    configs[entry_point.name] = config_class
                    _logger.info(f"Loaded exporter config: {entry_point.name}")
                except Exception as e:
                    _logger.warning(f"Failed to load exporter config {entry_point.name}: {e}")
        except Exception as e:
            _logger.warning(f"Failed to load exporter configurations: {e}")
        return configs

    def _get_package_configs(self) -> List[ExporterConfig]:
        """Get configurations from installed package configs."""
        package_configs = []

        for config_name, config_class in self.entry_point_exporter_configs.items():
            try:
                # Get default config from package
                default_config = config_class.get_default_config()
                exporter_config = ExporterConfig(
                    class_name=default_config["class_name"],
                    config=default_config.get("config", {}),
                    enabled=default_config.get("enabled", True),
                )
                package_configs.append(exporter_config)
            except Exception as e:
                _logger.warning(f"Failed to get default config for {config_name}: {e}")

        return package_configs

    def _get_file_config(self, config_file_path: Optional[str] = None) -> Optional[List[ExporterConfig]]:
        """Load configuration from file (config_file_path, cwd, or env)."""
        # Determine file path using priority order
        file_path = self._find_config_file(config_file_path)
        if not file_path:
            return None

        try:
            with open(file_path, "r") as f:
                if file_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    _logger.warning(f"Unsupported config file format: {file_path.suffix}")
                    return None

            # Convert to list of ExporterConfig
            if isinstance(config_data, dict) and "exporters" in config_data:
                # Handle the {"exporters": [...]} format from files
                return self._build_exporter_configs(config_data["exporters"])
            else:
                _logger.warning("Invalid config file format: expected dict with 'exporters' key")
                return None

        except Exception as e:
            _logger.warning(f"Failed to load config file {file_path}: {e}")
            return None

    def _find_config_file(self, config_file_path: Optional[str] = None) -> Optional[Path]:
        """Find configuration file using priority order."""
        # 1. Direct path provided
        if config_file_path:
            path = Path(config_file_path)
            if path.exists():
                return path

        # 2. Environment variable (higher priority than current working directory)
        env_path = os.getenv("ONE_LOGGER_EXPORTER_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path

        # 3. Current working directory
        cwd_files = [
            "one_logger_exporters_config.yaml",
            "one_logger_exporters_config.yml",
            "one_logger_exporters_config.json",
        ]

        for filename in cwd_files:
            path = Path.cwd() / filename
            if path.exists():
                return path

        return None

    def _merge_configs(self, base: List[ExporterConfig], override: List[ExporterConfig]) -> List[ExporterConfig]:
        """Merge configurations with override taking precedence."""
        # Merge exporters
        merged_exporters = {}

        # Add base exporters
        for exporter in base:
            merged_exporters[exporter.class_name] = exporter

        # Override with override exporters
        for exporter in override:
            if exporter.class_name in merged_exporters:
                _logger.info(f"Overriding exporter config for '{exporter.class_name}'")
            else:
                _logger.info(f"Adding new exporter config for '{exporter.class_name}'")
            merged_exporters[exporter.class_name] = exporter

        # Merge configs for same exporter types
        for class_name, exporter in merged_exporters.items():
            if class_name in [e.class_name for e in override]:
                # Deep merge configs
                base_exporter = next((e for e in base if e.class_name == class_name), None)
                if base_exporter:
                    merged_config = {**base_exporter.config, **exporter.config}
                    if merged_config != base_exporter.config:
                        _logger.info(f"Merged config for '{class_name}': {base_exporter.config} + {exporter.config} = {merged_config}")
                    exporter.config = merged_config

        return list(merged_exporters.values())

    def _build_exporter_configs(self, config_list: List[Dict[str, Any]]) -> List[ExporterConfig]:
        """Convert a list of dictionaries to a list of ExporterConfig."""
        exporters = []
        for config_dict in config_list:
            exporter_config = ExporterConfig(
                class_name=config_dict["class_name"],
                config=config_dict.get("config", {}),
                enabled=config_dict.get("enabled", True),
            )
            exporters.append(exporter_config)
        return exporters

    def _create_exporter_from_config(self, exporter_config: ExporterConfig, training_telemetry_config: Optional[Any] = None) -> Optional[Exporter]:
        """Create an exporter from configuration."""
        try:
            class_name = exporter_config.class_name
            config = exporter_config.config.copy()

            # Split the class name to get module and class
            if "." in class_name:
                module_name, class_name_part = class_name.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name_part])
                exporter_class = getattr(module, class_name_part)
            else:
                # Assume it's a class name in the current module
                exporter_class = globals()[class_name]

            # Handle special cases for specific exporter types
            if class_name == "nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter.V1CompatibleExporter":
                # V1CompatibleExporter needs one_logger_config and config dictionary
                if training_telemetry_config is None:
                    _logger.warning("V1CompatibleExporter requires one_logger_config but none was provided")
                    return None

                return exporter_class(one_logger_config=training_telemetry_config, config=config)
            elif "file_path" in config:
                # Convert string path to Path object for any exporter that uses file_path
                file_path_str = config.pop("file_path")
                config["file_path"] = Path(file_path_str)
                return exporter_class(**config)
            else:
                return exporter_class(**config)

        except (ImportError, AttributeError, KeyError) as e:
            _logger.warning(f"Failed to load exporter class {exporter_config.class_name}: {e}")
            return None
        except Exception as e:
            _logger.warning(f"Failed to create exporter {exporter_config.class_name}: {e}")
            return None
