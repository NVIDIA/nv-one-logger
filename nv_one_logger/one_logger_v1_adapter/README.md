# one_logger_v1_adapter
This project provides adapters for users of the previous version of one logger (v1) to transition to v2 smoothly.
In particular,

- It provides an adapter to convert v1 config to v2 config.
- It provides exporters that export telemetry data to the v1 backend (wandb) and ensures the metric names/format are the same so that the downstream consumers of telemetry data are not affected.
- It provides a unified `V1CompatibleExporter` factory class for creating v1-compatible exporters.

## V1CompatibleExporter API

The `V1CompatibleExporter` class provides a unified interface for creating v1-compatible wandb exporters that can work with either sync or async modes.

### Basic Usage

```python
from nv_one_logger.training_telemetry.v1_adapter import V1CompatibleExporter
from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

# Create base and training configs
base_config = OneLoggerConfig(
    application_name="my_app",
    session_tag_or_fn="my_session",
    world_size_or_fn=1,
)

training_config = TrainingTelemetryConfig(
    perf_tag_or_fn="my_perf_tag",
    # ... other training config parameters
)

# Create a sync exporter
exporter = V1CompatibleExporter(
    one_logger_config=base_config,
    async_mode=False  # Use sync mode
)

# Create an async exporter
exporter = V1CompatibleExporter(
    one_logger_config=base_config,
    async_mode=True  # Use async mode
)

# Configure the TrainingTelemetryProvider
TrainingTelemetryProvider.instance().with_base_config(base_config).with_exporter(exporter.exporter).configure_provider()

on_app_start <- api call
TrainingTelemetryProvider.instance().set_training_telemetry_config(training_config)
```

### Using configure_v2_adapter (Recommended)

For easier v1 to v2 migration, you can use the `configure_v2_adapter` function which handles all the configuration automatically:

```python
from nv_one_logger.training_telemetry.v1_adapter import configure_v2_adapter

# Define your v1-style configuration
v1_config = {
    "one_logger_project": "my_app",
    "app_tag": "my_perf_tag", 
    "app_tag_run_name": "my_session",
    "is_baseline_run": False,
    "world_size": 1,
    "global_batch_size": 32,
    "is_train_iterations_enabled": True,
    "is_validation_iterations_enabled": True,
    "one_logger_async": True,  # Use async mode
    "enable_for_current_rank": True,  # Enable for current rank
    "metadata": {
        "model_name": "gpt2",
        "dataset": "wikitext"
    }
}

# Configure OneLogger with v1 config - this handles everything automatically
configure_v2_adapter(v1_config)
```

The `configure_v2_adapter` function:
- Converts v1 config to v2 config using `ConfigAdapter`
- Creates a `V1CompatibleExporter` with proper tags
- Configures the `TrainingTelemetryProvider` singleton
- Handles async/sync mode based on `one_logger_async`
- Applies rank-based configuration via `enable_for_current_rank`

This is the recommended approach for applications migrating from v1 to v2 as it requires minimal code changes.

### Configuration

The `V1CompatibleExporter` automatically creates a `WandBConfig` with v1-compatible settings:

- **entity**: Always set to `"hwinf_dcm"` for internal users
- **project**: Uses `one_logger_config.application_name`
- **run_name**: Auto-generated with format `"{application_name}-run-{uuid}"`
- **host**: Set to `"https://api.wandb.ai"`
- **api_key**: Empty string (uses anonymous login)
- **tags**: Set to `["e2e_metrics_enabled"]`
- **save_dir**: Set to `"./wandb"`

### Properties

The `V1CompatibleExporter` provides the following properties:

- `exporter`: The underlying exporter instance (`V1CompatibleWandbExporterSync` or `V1CompatibleWandbExporterAsync`)
- `one_logger_config`: The OneLogger configuration
- `exporter_config`: The automatically created `WandBConfig`
- `is_async`: Boolean indicating if the exporter is in async mode

