# one_logger_v1_adapter
This project provides adapters for users of the previous version of one logger (v1) to transition to v2 smoothly.
In particular,

- It provides an adapter to convert v1 config to v2 config.
- It provides exporters that export telemetry data to the v1 backend (wandb) and ensures the metric names/format are the same so that the downstream consumers of telemetry data are not affected.