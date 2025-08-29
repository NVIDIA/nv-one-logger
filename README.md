# nv-one-logger repo

## Repo Structure

This repo is designed as a mono-repo for various nv-one-logger projects. Each project is independent (has its own CI/CD rules, is published as a separate docker image, etc).

This allows us to organize our code in different Python projects with narrow dependencies while making the development and code review process easier (as everything is in one repo).

## Project Folder Hierarchy

```
nv-one-logger-gitlab/
├── nv_one_logger/
│   ├── one_logger_core/                   # Core logging functionality
│   │   ├── src/nv_one_logger/             # Source code
│   │   │   ├── api/                       # Public API interfaces
│   │   │   ├── core/                      # Core implementations
│   │   │   ├── exporter/                  # Export mechanisms
│   │   │   └── recorder/                  # Recording implementations
│   │   ├── tests/                         # Unit tests
│   │   └── pyproject.toml                 # Python project configuration
│   │
│   ├── one_logger_otel/                   # OpenTelemetry integration
│   │   ├── src/nv_one_logger/otel/        # OTEL-specific implementations
│   │   │   └── exporter/                  # OTEL exporters
│   │   ├── tests/                         # OTEL-specific tests
│   │   └── pyproject.toml
│   │
│   ├── one_logger_training_telemetry/     # ML training telemetry
│   │   ├── src/nv_one_logger/training_telemetry/
│   │   │   ├── api/                       # Training telemetry API
│   │   │   └── docs/                      # Documentation and examples
│   │   ├── tests/                         # Training telemetry tests
│   │   └── pyproject.toml
│   │
│   ├── one_logger_pytorch_lightning_integration/ # PyTorch Lightning integration
│   │   ├── src/nv_one_logger/training_telemetry/
│   │   │   └── integration/               # PyTorch Lightning callbacks
│   │   ├── tests/                         # Integration tests
│   │   └── pyproject.toml
│   │
│   └── one_logger_wandb/                  # Weights & Biases integration
│       ├── src/nv_one_logger/wandb/       # WandB-specific implementations
│       │   └── exporter/                  # WandB exporters
│       ├── tests/                         # WandB-specific tests
│       └── pyproject.toml
│
├── third_party_licenses/                  # Third-party license files
└── README.md                              # This file
```

### Common Project Structure

Each project follows a consistent structure:
- `src/`: Source code organized under the `nv_one_logger` namespace
- `tests/`: Comprehensive test suites with unit and integration tests
- `pyproject.toml`: Poetry-based project configuration and dependencies
- `README.md`: Project-specific documentation
- `Makefile`: Automation scripts for common development tasks such as linting, formatting, testing, and publishing

### Project Descriptions

#### one_logger_core
The foundational logging library that provides:
- **API Layer**: Public interfaces for logging operations (`api/`)
- **Core Components**: Core logging functionality including spans, events, and attributes (`core/`)
- **Exporters**: Base classes and implementations for exporting telemetry data (`exporter/`)
- **Recorders**: Default recording implementations (`recorder/`)


#### Projects that facilitate telemetry in a particular domain

##### one_logger_training_telemetry
Machine learning training-specific telemetry package that:
- Provides specialized APIs for ML training workflows (`api/`)
- Includes training-specific attributes, events, and spans
- Offers callbacks and context management for training scenarios
- Contains examples for ML use cases (`docs/`)

#### Projects that facilitate using different telemetry backends

##### one_logger_otel
OpenTelemetry integration package that:
- Provides OTEL-compatible exporters for seamless integration with OpenTelemetry ecosystem
- Enables users to export telemetry data to OTEL collectors and compatible backends

##### one_logger_wandb
Weights & Biases integration package that:
- Provides exporters for sending telemetry data to WandB
- Supports both synchronous and asynchronous export modes
- Enables integration with WandB's experiment tracking and visualization tools

#### Application layer pacakges

##### one_logger_pytorch_lightning_integration
PyTorch Lightning integration package that:
- Provides ready-to-use callbacks for PyTorch Lightning training loops
- Enables automatic telemetry collection during PyTorch Lightning training
- Integrates seamlessly with the training telemetry system



## Dev Guidance

Note for NVIDIAN dev, please check the confluence page [Dev guidance on nv-one-logger](https://confluence.nvidia.com/display/MLWFO/Dev+guidance+on+nv-one-logger).
