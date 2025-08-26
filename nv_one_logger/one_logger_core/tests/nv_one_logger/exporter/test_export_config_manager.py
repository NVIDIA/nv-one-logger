# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from overrides import override

from nv_one_logger.core.event import Attributes, ErrorEvent, Event, TelemetryDataError
from nv_one_logger.core.span import Span
from nv_one_logger.exporter.export_config_manager import ExporterConfigManager
from nv_one_logger.exporter.exporter import Exporter
from nv_one_logger.exporter.exporter_config import ExporterConfig

# Ensure dynamic import strings like "test_export_config_manager.*" resolve to this module
sys.modules.setdefault("test_export_config_manager", sys.modules[__name__])


class TestFileExporter(Exporter):
    """Test file exporter for testing."""

    def __init__(self, file_path: str = "/tmp/test.log"):
        self.file_path = file_path
        self.initialized = False
        self.exported_data = []

    @override
    def initialize(self) -> None:
        """Initialize the test file exporter."""
        self.initialized = True
        # Create the file if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            f.write("")  # Create empty file

    @override
    def export_start(self, span: Span) -> None:
        """Export span start event."""
        self.exported_data.append({"type": "start", "data": span})
        with open(self.file_path, "a") as f:
            f.write(f"START: {span}\n")

    @override
    def export_stop(self, span: Span) -> None:
        """Export span stop event."""
        self.exported_data.append({"type": "stop", "data": span})
        with open(self.file_path, "a") as f:
            f.write(f"STOP: {span}\n")

    @override
    def export_event(self, event: Event, span: Span) -> None:
        """Export event."""
        self.exported_data.append({"type": "event", "data": event})
        with open(self.file_path, "a") as f:
            f.write(f"EVENT: {event}\n")

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Export error event."""
        self.exported_data.append({"type": "error", "data": event})
        with open(self.file_path, "a") as f:
            f.write(f"ERROR: {event}\n")

    @override
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Export telemetry data error."""
        self.exported_data.append({"type": "telemetry_error", "data": error})
        with open(self.file_path, "a") as f:
            f.write(f"TELEMETRY_ERROR: {error}\n")

    @override
    def close(self) -> None:
        """Close the exporter."""
        # Clean up if needed
        pass


class TestJsonExporter(Exporter):
    """Test JSON exporter for testing."""

    def __init__(self, output_file: str = "/tmp/test.json"):
        self.output_file = output_file
        self.initialized = False
        self.exported_data = []

    @override
    def initialize(self) -> None:
        """Initialize the test JSON exporter."""
        self.initialized = True
        # Create the file if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump([], f)  # Initialize with empty array

    @override
    def export_start(self, span: Span) -> None:
        """Export span start event."""
        self.exported_data.append({"type": "start", "data": span})
        self._write_data()

    @override
    def export_stop(self, span: Span) -> None:
        """Export span stop event."""
        self.exported_data.append({"type": "stop", "data": span})
        self._write_data()

    @override
    def export_event(self, event: Event, span: Span) -> None:
        """Export event."""
        self.exported_data.append({"type": "event", "data": event})
        self._write_data()

    @override
    def export_error(self, event: ErrorEvent, span: Span) -> None:
        """Export error event."""
        self.exported_data.append({"type": "error", "data": event})
        self._write_data()

    @override
    def export_telemetry_data_error(self, error: TelemetryDataError) -> None:
        """Export telemetry data error."""
        self.exported_data.append({"type": "telemetry_error", "data": error})
        self._write_data()

    @override
    def close(self) -> None:
        """Close the exporter."""
        # Clean up if needed
        pass

    def _write_data(self):
        """Write exported data to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.exported_data, f, indent=2)


class FileExporterConfig:
    """Test file exporter configuration class."""

    @staticmethod
    def get_default_config():
        """Get default configuration for file exporter."""
        return {
            "class_name": "test_export_config_manager.TestFileExporter",
            "config": {"file_path": "/tmp/default.log"},
            "enabled": True,
        }

    @staticmethod
    def create_exporter(config):
        """Create file exporter from configuration."""
        return TestFileExporter(config.get("file_path", "/tmp/default.log"))


class JsonExporterConfig:
    """Test JSON exporter configuration class."""

    @staticmethod
    def get_default_config():
        """Get default configuration for JSON exporter."""
        return {
            "class_name": "test_export_config_manager.TestJsonExporter",
            "config": {"output_file": "/tmp/default.json"},
            "enabled": True,
        }

    @staticmethod
    def create_exporter(config):
        """Create JSON exporter from configuration."""
        return TestJsonExporter(config.get("output_file", "/tmp/default.json"))


class TestExporterConfigManager:
    """Tests for ExporterConfigManager class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.manager = ExporterConfigManager()

    @pytest.mark.parametrize(
        "entry_points,expected_configs",
        [
            # Test single entry point
            (
                [
                    ("file_exporter", FileExporterConfig),
                ],
                {"file_exporter": FileExporterConfig},
            ),
            # Test multiple entry points
            (
                [
                    ("file_exporter", FileExporterConfig),
                    ("json_exporter", JsonExporterConfig),
                ],
                {
                    "file_exporter": FileExporterConfig,
                    "json_exporter": JsonExporterConfig,
                },
            ),
            # Test empty entry points
            ([], {}),
        ],
    )
    def test_init_loads_entry_points(self, entry_points, expected_configs):
        """Test that ExporterConfigManager loads entry points correctly during initialization."""
        with patch("nv_one_logger.exporter.export_config_manager.entry_points") as mock_entry_points:
            mock_entries = []
            for name, config_class in entry_points:
                mock_entry = Mock()
                mock_entry.name = name
                mock_entry.load.return_value = config_class
                mock_entries.append(mock_entry)

            # Mock entry_points() to return a dict-like object (Python 3.8-3.9 style)
            # Create a simple class that behaves like the real entry_points() result
            class MockEntryPoints:
                def get(self, group_name, default=None):
                    if group_name == "nv_one_logger.exporter_configs":
                        return mock_entries
                    return default or []

            mock_entry_points.return_value = MockEntryPoints()

            # Create a fresh manager instance with mocked entry points
            manager = ExporterConfigManager()

            assert manager.entry_point_exporter_configs == expected_configs

    @pytest.mark.parametrize(
        "direct_config,file_config,expected_exporters",
        [
            # No configs provided
            (None, None, []),
            # Only direct config
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/direct.log"},
                    )
                ],
                None,
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/direct.log"},
                    }
                ],
            ),
            # Only file config
            (
                None,
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestJsonExporter",
                            "config": {"output_file": "/tmp/file.json"},
                        }
                    ]
                },
                [
                    {
                        "class_name": "test_export_config_manager.TestJsonExporter",
                        "config": {"output_file": "/tmp/file.json"},
                    }
                ],
            ),
            # Direct config overrides file config
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/override.log"},
                    )
                ],
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestFileExporter",
                            "config": {"file_path": "/tmp/file.log"},
                        }
                    ]
                },
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/override.log"},
                    }
                ],
            ),
            # Merge different exporter types
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/direct.log"},
                    )
                ],
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestJsonExporter",
                            "config": {"output_file": "/tmp/file.json"},
                        }
                    ]
                },
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/direct.log"},
                    },
                    {
                        "class_name": "test_export_config_manager.TestJsonExporter",
                        "config": {"output_file": "/tmp/file.json"},
                    },
                ],
            ),
            # Direct config adds new fields to file config (merge behavior)
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/direct.log", "new_param": "new_value"},
                    )
                ],
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestFileExporter",
                            "config": {"file_path": "/tmp/base.log", "existing_param": "existing_value"},
                        }
                    ]
                },
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/direct.log", "existing_param": "existing_value", "new_param": "new_value"},
                    }
                ],
            ),
        ],
    )
    def test_generate_export_config(self, direct_config, file_config, expected_exporters):
        """Test generating export configuration with priority-based merging."""
        # Mock entry points to return empty list to avoid loading package configs
        with patch(
            "nv_one_logger.exporter.export_config_manager.entry_points",
            return_value=[],
        ):
            # Create manager instance
            manager = ExporterConfigManager()

            # Create temporary file if file_config provided
            temp_file = None
            if file_config:
                temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                yaml.dump(file_config, temp_file)
                temp_file.close()

            try:
                config = manager.generate_export_config(
                    direct_config=direct_config,
                    config_file_path=temp_file.name if temp_file else None,
                )

                assert isinstance(config, list)
                assert len(config) == len(expected_exporters)

                # Check that all expected exporters are present (order doesn't matter)
                actual_exporters = {(e.class_name, tuple(sorted(e.config.items()))) for e in config}
                expected_exporters_set = {(e["class_name"], tuple(sorted(e["config"].items()))) for e in expected_exporters}
                assert actual_exporters == expected_exporters_set

            finally:
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    @pytest.mark.parametrize(
        "package_configs,file_config,direct_config,expected_exporters",
        [
            # File config completely replaces package configs
            (
                {"file_exporter": FileExporterConfig},  # Package provides FileExporter
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestJsonExporter",
                            "config": {"output_file": "/tmp/override.json"},
                        }
                    ]
                },
                None,
                [
                    {
                        "class_name": "test_export_config_manager.TestJsonExporter",
                        "config": {"output_file": "/tmp/override.json"},
                    }
                ],
            ),
            # Direct config merges with package configs (file config is None)
            (
                {"file_exporter": FileExporterConfig},
                None,
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestJsonExporter",
                        config={"output_file": "/tmp/direct.json"},
                    )
                ],
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/default.log"},
                    },
                    {
                        "class_name": "test_export_config_manager.TestJsonExporter",
                        "config": {"output_file": "/tmp/direct.json"},
                    },
                ],
            ),
            # Direct config overrides file config, both override package config
            (
                {"file_exporter": FileExporterConfig},
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestFileExporter",
                            "config": {"file_path": "/tmp/file_override.log"},
                        }
                    ]
                },
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/direct_override.log", "new_param": "value"},
                    )
                ],
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/direct_override.log", "new_param": "value"},
                    }
                ],
            ),
        ],
    )
    def test_generate_export_config_with_package_override(self, package_configs, file_config, direct_config, expected_exporters):
        """Test that direct/file configs properly override package configs."""
        # Mock entry points to return our test package configs
        mock_entries = []
        for name, config_class in package_configs.items():
            mock_entry = Mock()
            mock_entry.name = name
            mock_entry.load.return_value = config_class
            mock_entries.append(mock_entry)

        # Mock entry_points() to return a dict-like object (Python 3.8-3.9 style)
        # Create a simple class that behaves like the real entry_points() result
        class MockEntryPoints:
            def get(self, group_name, default=None):
                if group_name == "nv_one_logger.exporter_configs":
                    return mock_entries
                return default or []

        with patch(
            "nv_one_logger.exporter.export_config_manager.entry_points",
            return_value=MockEntryPoints(),
        ):
            # Create manager instance with package configs
            manager = ExporterConfigManager()

            # Create temporary file if file_config provided
            temp_file = None
            if file_config:
                temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                yaml.dump(file_config, temp_file)
                temp_file.close()

            try:
                config = manager.generate_export_config(
                    direct_config=direct_config,
                    config_file_path=temp_file.name if temp_file else None,
                )

                assert isinstance(config, list)
                assert len(config) == len(expected_exporters)

                # Check that all expected exporters are present (order doesn't matter for different types)
                actual_exporters = {(e.class_name, tuple(sorted(e.config.items()))) for e in config}
                expected_exporters_set = {(e["class_name"], tuple(sorted(e["config"].items()))) for e in expected_exporters}
                assert actual_exporters == expected_exporters_set

            finally:
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    @pytest.mark.parametrize(
        "config_file_path,env_var,expected_found",
        [
            # Direct path provided and exists
            ("/tmp/exists.yaml", None, True),
            # Direct path provided but doesn't exist
            ("/tmp/nonexistent.yaml", None, False),
            # No direct path, file in CWD
            (None, None, True),  # Will create file in CWD
            # Environment variable set
            (None, "/tmp/env.yaml", True),
            # No config found
            (None, None, False),  # When no files exist
        ],
    )
    def test_find_config_file(self, config_file_path, env_var, expected_found):
        """Test finding configuration file using priority order."""
        # Set up environment variable if provided
        if env_var:
            with patch.dict(os.environ, {"ONE_LOGGER_EXPORTER_CONFIG_PATH": env_var}):
                self._test_find_config_file_impl(config_file_path, expected_found)
        else:
            self._test_find_config_file_impl(config_file_path, expected_found)

    def _test_find_config_file_impl(self, config_file_path, expected_found):
        """Implement the test_find_config_file test logic."""
        # Track created files for cleanup
        created_files = []

        # Create test files if needed
        if config_file_path and expected_found:
            Path(config_file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(config_file_path).touch()
            created_files.append(Path(config_file_path))

        if expected_found and not config_file_path:
            # Create a file in CWD
            cwd_file = Path.cwd() / "one_logger_exporters_config.yaml"
            cwd_file.touch()
            created_files.append(cwd_file)

        try:
            # Test the method
            with patch("nv_one_logger.exporter.export_config_manager.entry_points") as mock_entry_points:
                mock_entry_points.return_value = []
                manager = ExporterConfigManager()
                result = manager._find_config_file(config_file_path)

            if expected_found:
                assert result is not None
                assert result.exists()
            else:
                assert result is None
        finally:
            # Clean up all created files
            for file_path in created_files:
                file_path.unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "file_content,file_format,expected_config",
        [
            # YAML file
            (
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestFileExporter",
                            "config": {"file_path": "/tmp/test.log"},
                            "enabled": True,
                        }
                    ]
                },
                "yaml",
                {
                    "exporter_count": 1,
                    "class_names": ["test_export_config_manager.TestFileExporter"],
                    "configs": [{"file_path": "/tmp/test.log"}],
                    "enabled": [True],
                },
            ),
            # JSON file
            (
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestJsonExporter",
                            "config": {"output_file": "/tmp/test.json"},
                        }
                    ]
                },
                "json",
                {
                    "exporter_count": 1,
                    "class_names": ["test_export_config_manager.TestJsonExporter"],
                    "configs": [{"output_file": "/tmp/test.json"}],
                    "enabled": [True],
                },
            ),
            # Multiple exporters (YAML)
            (
                {
                    "exporters": [
                        {
                            "class_name": "test_export_config_manager.TestFileExporter",
                            "config": {"file_path": "/tmp/1.log"},
                        },
                        {
                            "class_name": "test_export_config_manager.TestJsonExporter",
                            "config": {"output_file": "/tmp/1.json"},
                        },
                        {
                            "class_name": "test_export_config_manager.TestFileExporter",
                            "config": {"file_path": "/tmp/2.log"},
                            "enabled": False,
                        },
                    ]
                },
                "yaml",
                {
                    "exporter_count": 3,
                    "class_names": [
                        "test_export_config_manager.TestFileExporter",
                        "test_export_config_manager.TestJsonExporter",
                        "test_export_config_manager.TestFileExporter",
                    ],
                    "configs": [
                        {"file_path": "/tmp/1.log"},
                        {"output_file": "/tmp/1.json"},
                        {"file_path": "/tmp/2.log"},
                    ],
                    "enabled": [True, True, False],
                },
            ),
        ],
    )
    def test_get_file_config(self, file_content, file_format, expected_config):
        """Test loading configuration from YAML and JSON files."""
        # Create temporary config file with appropriate format
        suffix = f".{file_format}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            if file_format == "yaml":
                yaml.dump(file_content, f)
            elif file_format == "json":
                json.dump(file_content, f)
            config_file = f.name

        try:
            # Mock _find_config_file to return our test file
            with patch.object(self.manager, "_find_config_file", return_value=Path(config_file)):
                result = self.manager._get_file_config()

            assert result is not None
            assert len(result) == expected_config["exporter_count"]

            for i, exporter in enumerate(result):
                assert exporter.class_name == expected_config["class_names"][i]
                assert exporter.config == expected_config["configs"][i]
                assert exporter.enabled == expected_config["enabled"][i]

        finally:
            # Clean up
            Path(config_file).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "base_config,override_config,expected_result",
        [
            # Simple override
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/base.log"},
                    )
                ],
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/override.log"},
                    )
                ],
                {"exporter_count": 1, "configs": [{"file_path": "/tmp/override.log"}]},
            ),
            # Add new exporter
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/base.log"},
                    )
                ],
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestJsonExporter",
                        config={"output_file": "/tmp/new.json"},
                    )
                ],
                {
                    "exporter_count": 2,
                    "configs": [
                        {"file_path": "/tmp/base.log"},
                        {"output_file": "/tmp/new.json"},
                    ],
                },
            ),
            # Merge configs for same exporter type
            (
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/base.log", "param1": "value1"},
                    )
                ],
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={
                            "file_path": "/tmp/override.log",
                            "param2": "value2",
                        },
                    )
                ],
                {
                    "exporter_count": 1,
                    "configs": [
                        {
                            "file_path": "/tmp/override.log",
                            "param1": "value1",
                            "param2": "value2",
                        }
                    ],
                },
            ),
        ],
    )
    def test_merge_configs(self, base_config, override_config, expected_result):
        """Test merging configurations with override taking precedence."""
        result = self.manager._merge_configs(base_config, override_config)

        assert len(result) == expected_result["exporter_count"]

        for i, exporter in enumerate(result):
            assert exporter.config == expected_result["configs"][i]

    @pytest.mark.parametrize(
        "exporter_configs,config,expected_count,expected_exporters",
        [
            # Registered exporter
            (
                {"file_exporter": FileExporterConfig},
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/test.log"},
                    )
                ],
                1,
                [{"type": TestFileExporter, "file_path": "/tmp/test.log"}],
            ),
            # Custom exporter (using unified format)
            (
                {},
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/custom.log"},
                    )
                ],
                1,
                [{"type": TestFileExporter, "file_path": "/tmp/custom.log"}],
            ),
            # Mixed exporters
            (
                {
                    "file_exporter": FileExporterConfig,
                    "json_exporter": JsonExporterConfig,
                },
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/test.log"},
                    ),
                    ExporterConfig(
                        class_name="test_export_config_manager.TestJsonExporter",
                        config={"output_file": "/tmp/test.json"},
                    ),
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/custom.log"},
                    ),
                ],
                3,
                [
                    {"type": TestFileExporter, "file_path": "/tmp/test.log"},
                    {"type": TestJsonExporter, "output_file": "/tmp/test.json"},
                    {"type": TestFileExporter, "file_path": "/tmp/custom.log"},
                ],
            ),
            # Disabled exporter
            (
                {"file_exporter": FileExporterConfig},
                [
                    ExporterConfig(
                        class_name="test_export_config_manager.TestFileExporter",
                        config={"file_path": "/tmp/test.log"},
                        enabled=False,
                    )
                ],
                0,
                [],
            ),
            # Unknown exporter type
            (
                {},
                [ExporterConfig(class_name="unknown.module.UnknownExporter", config={})],
                0,
                [],
            ),
        ],
    )
    def test_create_exporters_from_config(self, exporter_configs, config, expected_count, expected_exporters):
        """Test creating exporter instances from configuration."""
        # Create manager instance
        manager = ExporterConfigManager()

        # Add exporter configs
        manager.entry_point_exporter_configs.update(exporter_configs)

        exporters = manager.create_exporters_from_config(config)

        assert len(exporters) == expected_count
        assert all(isinstance(exporter, Exporter) for exporter in exporters)

        # Check exporter types and their specific attributes
        for i, expected_exporter in enumerate(expected_exporters):
            assert isinstance(exporters[i], expected_exporter["type"])

            # Check exporter-specific attributes
            if expected_exporter["type"] == TestFileExporter:
                # file_path is converted to Path object, so compare string representation
                assert str(exporters[i].file_path) == expected_exporter["file_path"]
            elif expected_exporter["type"] == TestJsonExporter:
                assert exporters[i].output_file == expected_exporter["output_file"]

    def test_real_exporter_functionality(self):
        """Test that created exporters actually work with real telemetry data."""
        # Create a simple configuration
        config = [
            ExporterConfig(
                class_name="test_export_config_manager.TestFileExporter",
                config={"file_path": "/tmp/real_test.log"},
            )
        ]

        # Create exporters
        exporters = self.manager.create_exporters_from_config(config)
        assert len(exporters) == 1
        exporter = exporters[0]
        assert isinstance(exporter, TestFileExporter)

        # Test exporter functionality
        exporter.initialize()
        assert exporter.initialized

        # Create a test span
        span = Span.create("test_span")

        # Test export methods
        exporter.export_start(span)
        assert len(exporter.exported_data) == 1
        assert exporter.exported_data[0]["type"] == "start"

        # Test event export
        event = Event.create("test_event", attributes=Attributes({"key": "value"}))
        exporter.export_event(event, span)
        assert len(exporter.exported_data) == 2
        assert exporter.exported_data[1]["type"] == "event"

        # Test error export
        error_event = ErrorEvent.create("test_error", exception=Exception("test error"))
        exporter.export_error(error_event, span)
        assert len(exporter.exported_data) == 3
        assert exporter.exported_data[2]["type"] == "error"

        # Test stop export
        span.stop()
        exporter.export_stop(span)
        assert len(exporter.exported_data) == 4
        assert exporter.exported_data[3]["type"] == "stop"

        # Clean up
        exporter.close()
        Path("/tmp/real_test.log").unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "input_dict,expected_config",
        [
            # Basic dictionary with single exporter
            (
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/test.log"},
                        "enabled": True,
                    }
                ],
                {
                    "exporter_count": 1,
                    "class_names": ["test_export_config_manager.TestFileExporter"],
                    "configs": [{"file_path": "/tmp/test.log"}],
                    "enabled": [True],
                },
            ),
            # Dictionary with multiple exporters
            (
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/file1.log"},
                        "enabled": True,
                    },
                    {
                        "class_name": "test_export_config_manager.TestJsonExporter",
                        "config": {"output_file": "/tmp/file2.json"},
                        "enabled": False,
                    },
                ],
                {
                    "exporter_count": 2,
                    "class_names": [
                        "test_export_config_manager.TestFileExporter",
                        "test_export_config_manager.TestJsonExporter",
                    ],
                    "configs": [
                        {"file_path": "/tmp/file1.log"},
                        {"output_file": "/tmp/file2.json"},
                    ],
                    "enabled": [True, False],
                },
            ),
            # Dictionary with default enabled value
            (
                [
                    {
                        "class_name": "test_export_config_manager.TestFileExporter",
                        "config": {"file_path": "/tmp/test.log"},
                        # enabled defaults to True
                    }
                ],
                {
                    "exporter_count": 1,
                    "class_names": ["test_export_config_manager.TestFileExporter"],
                    "configs": [{"file_path": "/tmp/test.log"}],
                    "enabled": [True],
                },
            ),
            # Empty exporters list
            (
                [],
                {"exporter_count": 0, "class_names": [], "configs": [], "enabled": []},
            ),
        ],
    )
    def test_build_exporter_configs(self, input_dict, expected_config):
        """Test converting list of dictionaries to ExporterConfig objects."""
        result = self.manager._build_exporter_configs(input_dict)

        assert len(result) == expected_config["exporter_count"]

        for i, exporter in enumerate(result):
            assert exporter.class_name == expected_config["class_names"][i]
            assert exporter.config == expected_config["configs"][i]
            assert exporter.enabled == expected_config["enabled"][i]

    def test_generate_export_config_with_list_of_dicts(self):
        """Test generating export configuration with list of dictionaries input."""
        # Test with list of dictionaries input
        list_of_dicts_config = [
            {
                "class_name": "test_export_config_manager.TestFileExporter",
                "config": {"file_path": "/tmp/list_test1.log"},
                "enabled": True,
            },
            {
                "class_name": "test_export_config_manager.TestJsonExporter",
                "config": {"output_file": "/tmp/list_test2.json"},
                "enabled": False,
            },
        ]

        with patch("nv_one_logger.exporter.export_config_manager.entry_points") as mock_entry_points:
            mock_entry_points.return_value = []
            manager = ExporterConfigManager()
            result = manager.generate_export_config(list_of_dicts_config)

        assert len(result) == 2
        assert result[0].class_name == "test_export_config_manager.TestFileExporter"
        assert result[0].config == {"file_path": "/tmp/list_test1.log"}
        assert result[0].enabled
        assert result[1].class_name == "test_export_config_manager.TestJsonExporter"
        assert result[1].config == {"output_file": "/tmp/list_test2.json"}
        assert not result[1].enabled


class TestExporterConfig:
    """Test ExporterConfig dataclass."""

    @pytest.mark.parametrize(
        "class_name,config,enabled,expected",
        [
            (
                "test_export_config_manager.TestFileExporter",
                {},
                True,
                {
                    "class_name": "test_export_config_manager.TestFileExporter",
                    "config": {},
                    "enabled": True,
                },
            ),
            (
                "test_export_config_manager.TestJsonExporter",
                {"output_file": "/tmp/test.json"},
                False,
                {
                    "class_name": "test_export_config_manager.TestJsonExporter",
                    "config": {"output_file": "/tmp/test.json"},
                    "enabled": False,
                },
            ),
        ],
    )
    def test_exporter_config(self, class_name, config, enabled, expected):
        """Test ExporterConfig dataclass creation and attributes."""
        exporter_config = ExporterConfig(class_name=class_name, config=config, enabled=enabled)

        assert exporter_config.class_name == expected["class_name"]
        assert exporter_config.config == expected["config"]
        assert exporter_config.enabled == expected["enabled"]
