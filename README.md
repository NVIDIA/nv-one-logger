# nv-one-logger repo

## Repo Structure

This repo is designed as a mono-repo for various nv-one-logger projects. Each project is independent (has its own CI/Cd rules, is published as a separate docker image, etc).

This allows us to organize our code in different Python projects with narrow dependencies while making the development and code review process easier (as everything is in one repo).

## Projects

- one-logger-core

Projects that facilitate telemetry in a particular domain:

- one-logger-training-telemetry

Projects that facilitate using different telemetry backends:

- one-logger-wandb
- one-logger-otel
