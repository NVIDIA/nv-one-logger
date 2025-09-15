# Exporters (nv_one_logger.exporter)

This folder contains the exporter interface, base classes, built‑in exporters, and a configuration manager (recommended) to wire exporters via files or entry points.

- Interfaces and bases: `exporter.py`, `base_json_exporter.py`
- Built‑in exporters: `logger_exporter.py`, `file_exporter.py`
- Configuration utilities: `exporter_config.py`, `export_config_manager.py`

## Quick start

```python
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.exporter.file_exporter import FileExporter
from nv_one_logger.exporter.logger_exporter import LoggerExporter
from pathlib import Path
import logging

provider = TrainingTelemetryProvider.instance()
(
    provider
    .with_base_config(TrainingTelemetryConfig(application_name="demo", session_tag_or_fn="run1"))
    .with_exporter(FileExporter(file_path=Path("/tmp/one_logger_metrics.jsonl")))
    .with_exporter(LoggerExporter(logger=logging.getLogger("my_app.metrics")))
    .configure_provider()
)
```

---

## Interfaces and base classes

### Exporter (protocol) — `exporter.py`
Core contract all exporters implement:
- `initialize()` — prepare resources
- `export_start(span)` — span started
- `export_stop(span)` — span finished
- `export_event(event, span)` — custom event
- `export_error(error_event, span)` — application error
- `export_telemetry_data_error(error)` — schema/validation issue
- `close()` — release resources

Raise `ExportError` for recoverable export failures.

### BaseExporter — `exporter.py`
- Guards correct lifecycle/state: not‑initialized → ready → closed
- Validates usage in all export methods

### BaseJsonExporter — `base_json_exporter.py`
- Converts spans/events to JSON‑compatible dicts
- Adds `count` (incremental record id) and `pid`
- Emits record types via `RecordType` (`internal/record_type.py`)
- Optimizes single‑span case by emitting a single `complete` record when possible
- Delegates the actual write via `_write(json_dict)` (implemented by concrete exporters)

`RecordType` values: `START`, `STOP`, `COMPLETE`, `EVENT`, `APPLICATION_ERROR`, `TELEMETRY_DATA_ERROR`.

---

## Built‑in exporters

### LoggerExporter — `logger_exporter.py`
Emits formatted key=value logs to a provided Python `logging.Logger`.

Constructor:
```python
LoggerExporter(logger: logging.Logger)
```

Usage:
```python
import logging
from nv_one_logger.exporter.logger_exporter import LoggerExporter

exporter = LoggerExporter(logger=logging.getLogger("my_app.metrics"))
exporter.initialize()
```

Notes:
- Designed for immediate, human‑readable logging. If you need structured lines for ingestion, prefer `FileExporter`.
- The configuration manager cannot construct a `logging.Logger` from a string. Use programmatic wiring (`with_exporter`) for `LoggerExporter`.

### FileExporter — `file_exporter.py`
Writes each record as a JSON line to a file.

Constructor:
```python
FileExporter(file_path: pathlib.Path)
```

Behavior:
- Creates parent directories if missing
- Opens file on `initialize()` and writes JSON Lines
- Thread‑safe writes via an internal lock

Usage:
```python
from pathlib import Path
from nv_one_logger.exporter.file_exporter import FileExporter

exporter = FileExporter(file_path=Path("/var/log/one_logger.jsonl"))
exporter.initialize()
```

---

## Using exporters with the provider

```python
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.exporter.file_exporter import FileExporter
from nv_one_logger.exporter.logger_exporter import LoggerExporter
import logging
from pathlib import Path

provider = TrainingTelemetryProvider.instance()
(
    provider
    .with_base_config(TrainingTelemetryConfig(application_name="demo", session_tag_or_fn="run1"))
    .with_exporter(FileExporter(file_path=Path("/tmp/one_logger.jsonl")))
    .with_exporter(LoggerExporter(logger=logging.getLogger("my_app.metrics")))
    .configure_provider()
)
```

---

## Export configuration manager (Recommended)

### Overview
Use `with_export_config(...)` to declaratively configure exporters from:
- Direct config (in code)
- Configuration files (YAML/JSON)
- Package‑provided configs via Python entry points

This enables teams to ship defaults and applications to override safely.

### Configuration sources
- Direct configuration in code (highest priority)
- File configuration (medium priority)
- Package configuration via entry points (lowest priority)

### Priority rules
- Same exporter `class_name`: higher priority replaces lower; config is shallow‑merged
- Different exporter classes: combined

### File discovery
1. Explicit path via `with_export_config(config_file_path=...)`
2. `ONE_LOGGER_EXPORTER_CONFIG_PATH` environment variable
3. CWD files: `one_logger_exporters_config.yaml|yml|json`

Set via environment:
```bash
export ONE_LOGGER_EXPORTER_CONFIG_PATH="/path/to/one_logger_exporters_config.yaml"
```

### Examples

Basic usage with package configs and discovery:
```python
provider = TrainingTelemetryProvider.instance()
provider.with_export_config().configure_provider()
```

Direct configuration (highest priority):
```python
provider = TrainingTelemetryProvider.instance()
provider.with_export_config(exporters_config=[
  {
    "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
    "config": {"file_path": "/tmp/direct.log"},
    "enabled": True
  }
]).configure_provider()
```

Replace package configs with file:
```yaml
# one_logger_exporters_config.yaml
exporters:
  - class_name: "nv_one_logger.exporter.file_exporter.FileExporter"
    config:
      file_path: "/tmp/override.log"
```
```python
provider = TrainingTelemetryProvider.instance()
provider.with_export_config(config_file_path="one_logger_exporters_config.yaml").configure_provider()
```

Combine multiple sources (direct overrides file):
```python
provider = TrainingTelemetryProvider.instance()
provider.with_export_config(
  exporters_config=[{"class_name": "nv_one_logger.exporter.file_exporter.FileExporter", "config": {"file_path": "/tmp/override.log"}}],
  config_file_path="one_logger_exporters_config.yaml",
).configure_provider()
```

Notes:
- Use config manager for exporters that accept serializable constructor args (e.g., `FileExporter`).
- `LoggerExporter` requires a `logging.Logger`; add it programmatically with `.with_exporter(LoggerExporter(logger=...))`.

### Data model — `exporter_config.py`
```python
@dataclass
class ExporterConfig:
    class_name: str              # e.g. "nv_one_logger.exporter.file_exporter.FileExporter"
    config: Dict[str, Any] = {}  # kwargs passed to the exporter constructor
    enabled: bool = True
```

### Sources and priority
Merged in the following order (higher wins on conflicts):
1. Direct config passed to `with_export_config(exporters_config=...)`
2. File config (YAML/JSON)
3. Package configs discovered via entry points

Merging rules:
- Same `class_name`: higher priority replaces lower (config is shallow‑merged for convenience)
- Different `class_name`s: all included

### File format
The file must be a dict with an `exporters` array.

YAML:
```yaml
exporters:
  - class_name: "nv_one_logger.exporter.file_exporter.FileExporter"
    config:
      file_path: "/tmp/app.log"   # str is accepted; manager converts to Path
    enabled: true
```

JSON:
```json
{
  "exporters": [
    {
      "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
      "config": { "file_path": "/tmp/app.log" },
      "enabled": true
    }
  ]
}
```

File discovery order:
1. Explicit path passed to `with_export_config(config_file_path=...)`
2. Environment variable `ONE_LOGGER_EXPORTER_CONFIG_PATH`
3. Current working directory files: `one_logger_exporters_config.yaml|yml|json`

### Package‑provided configs (entry points)
Packages can expose default exporter configs using the entry point group `nv_one_logger.exporter_configs`.

**Using setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="my-team-exporter-config",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "nv_one_logger.exporter_configs": [
            "my_file_exporter = my_team_exporter_config.configs:MyFileExporterConfig",
        ],
    },
    install_requires=["nv-one-logger-core"],
)
```

**Using pyproject.toml (Poetry):**
```toml
[tool.poetry]
name = "my-team-exporter-config"
version = "0.1.0"
description = "My team's exporter configuration"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.8"
nv-one-logger-core = "^2.1.0-rc.2"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.plugins."nv_one_logger.exporter_configs"]
my_file_exporter = "my_team_exporter_config.configs:MyFileExporterConfig"
```

`configs.py`:
```python
class MyFileExporterConfig:
    @staticmethod
    def get_default_config():
        return {
            "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
            "config": {"file_path": "/tmp/my_team_app.log"},
            "enabled": True,
        }
```

Notes:
- Only exporters whose constructor arguments are serializable should be configured via files/entry points. `FileExporter` is supported. `LoggerExporter` requires a `logging.Logger` and should be added programmatically with `.with_exporter(...)`.

### API usage via the provider
```python
provider = TrainingTelemetryProvider.instance()
# File or env discovery
provider.with_export_config().configure_provider()

# Explicit file path
provider.with_export_config(config_file_path="one_logger_exporters_config.yaml").configure_provider()

# Direct list of dicts
provider.with_export_config(exporters_config=[
  {
    "class_name": "nv_one_logger.exporter.file_exporter.FileExporter",
    "config": {"file_path": "/tmp/direct.log"},
    "enabled": True
  }
]).configure_provider()

# Combine direct + file (direct overrides)
provider.with_export_config(
  exporters_config=[{"class_name": "nv_one_logger.exporter.file_exporter.FileExporter", "config": {"file_path": "/tmp/override.log"}}],
  config_file_path="one_logger_exporters_config.yaml",
).configure_provider()
```

---

## Error handling
- Exporters should raise `nv_one_logger.exporter.exporter.ExportError` on export failures
- Always call `close()` to flush/release resources
- `BaseJsonExporter` will log a warning if a span was started but never stopped at `close()`

---

## Reference
- `nv_one_logger.exporter.exporter`: `ExportError`, `Exporter`, `BaseExporter`
- `nv_one_logger.exporter.base_json_exporter`: JSON serialization helpers and `_write(...)` hook
- `nv_one_logger.exporter.logger_exporter.LoggerExporter`: logs as key=value strings
- `nv_one_logger.exporter.file_exporter.FileExporter`: writes JSON Lines to file
- `nv_one_logger.exporter.internal.record_type.RecordType`: record categories
- `nv_one_logger.exporter.exporter_config.ExporterConfig`: data model
- `nv_one_logger.exporter.export_config_manager.ExporterConfigManager`: priority‑based config loader and builder 