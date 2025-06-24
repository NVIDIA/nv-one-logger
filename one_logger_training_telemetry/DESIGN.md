- Relationship between callback and TrainingTelemetry

- provider singleton
To make training telemtry work, we need a few ingredients:
- a config of type `TrainingTelemetryConfig`
- a list of exporters that determine which telemetry backend(s) are used.
- A recorder of type TrainingRecorder

To make using context managers and callbacks easier, we have defined a singleton called `TrainingTelemetryProvider`. The user is expected to
configure this singleton before using the library:

```python
TrainingTelemetryProvider.instance().configure(config=..., exporters=[...])
```

After this call, you can use the context managers, the callbacks, or directly access the recorder. See README.md for more info on that.


- tests with singleton?



- Relationship with the core project
- Predefined spans and events
- Attributes and class preservation
- We decided not to add attributes after the fact Example of ckpt span
- Mandatory and optional fields in attr classes
