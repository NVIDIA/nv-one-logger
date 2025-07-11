# One Logger Training Telemetry

One Logger Training Telemetry is built on top of one-logger-core using the latter to collect telemetry data on training jobs. It includes

- Predefined spans, events, and attributes for a typical training job.
- Easy integration with several trainng frameworks.

## Concepts

Similar to one-logger-core, this library uses concepts inspired by Open Telemetry ([spans](https://opentelemetry.io/docs/specs/otel/overview/#spans), [events](https://opentelemetry.io/docs/concepts/signals/traces/#span-events), and [attributes](https://opentelemetry.io/docs/specs/otel/common/#attribute)). 
To illustrate the concepts better, the table below shows an example
structure (parent-child span relationships) in a *typical* training job with synchronous checkpoint saving(each cell of the table represents a span, which is a child of the span shown in the cell left of it). Note that this is just an example;
the exact structure of spans is determined by the actual structure of code, which is reflected in the timing and order in which the training job calls the telemetry callbacks (or context managers).

<!-- DO NOT CONVERT THIS TABLE TO A MARKDOWN TABLE. It has cells that arte merged vertically, which is not something markdown supports. -->


<table style="border: 1px solid black;">
        <tr style="border: 1px solid black;">
            <td rowspan=12 style="border: 1px solid black; background-color:rgb(215, 245, 246)">APPLICATION span</td>
            <td style="border: 1px solid black;">DIST_INIT span</td>
            <td style="border: 1px solid black;"></td>
            <td style="border: 1px solid black;"></td>
        </tr>
        <tr style="border: 1px solid black;">
        <td style="border: 1px solid black;">DATA_LOADER_INIT span</td>
        <td style="border: 1px solid black;"></td>
        <td style="border: 1px solid black;"></td>
        </tr>
        <tr style="border: 1px solid black; ">           
            <td style="border: 1px solid black;">CHECKPOINT_LOAD span</td>
            <td style="border: 1px solid black;"></td>
            <td style="border: 1px solid black;"></td>
        <tr style="border: 1px solid black; ">
            <td style="border: 1px solid black;">MODEL_INIT span</td>
            <td style="border: 1px solid black;"></td>
            <td style="border: 1px solid black;"></td>
        <tr style="border: 1px solid black; ">
            <td style="border: 1px solid black;">OPTIMIZER_INIT span</td>
            <td style="border: 1px solid black;"></td>
            <td style="border: 1px solid black;"></td>
        </tr>
        <tr style="border: 1px solid black; ">
            <td rowspan=6 style="border: 1px solid black; background-color:rgb(243, 248, 203)">TRAINING_LOOP span</td>
            <td rowspan=5 style="border: 1px solid black; background-color:rgb(243, 248, 203)">TRAINING_SINGLE_ITERATION span*</td>
            <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">DATA_LOADING span</td>
        </tr>
        <tr style="border: 1px solid black; ">
        <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">MODEL_FORWARD span</td>
        </tr>
        <tr style="border: 1px solid black; ">
        <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">MODEL_BACKWARD span</td>
        </tr>
        <tr style="border: 1px solid black; ">
        <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">OPTIMIZER_UPDATE</td>
        </tr>
        <tr style="border: 1px solid black;">
            <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">CHECKPOINT_SAVE_SYNC span*</td>
        </tr>
        <tr style="border: 1px solid black; ">
        <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">VALIDATION_LOOP span*</td>
        <td style="border: 1px solid black; background-color:rgb(243, 248, 203)">VALIDATION_SINGLE_ITERATION span</td>
        </tr>
        <tr style="border: 1px solid black; ">
            <td rowspan=1 style="border: 1px solid black; background-color:rgb(244, 203, 248)">TESTING_LOOP span</td>
            <td style="border: 1px solid black; background-color:rgb(244, 203, 248)">TESTING_SINGLE_ITERATION span </td>
            <td style="border: 1px solid black; background-color:rgb(244, 203, 248)"></td>
        </tr>
</table>


> **_NOTE:_** 
"*" means that the parent span can have multiple instances of this span. For example, training loop has multiple spans of type TRAINING_SINGLE_ITERATION (one for each iteration of training).

<!-- DO NOT CONVERT THE ABOVE TABLE TO A MARKDOWN TABLE. It has cells that are merged vertically, which is not something markdown supports. -->

 That is, the application (represented as the _application span_) includes a training loop and an optional testing loop (each represented as a child span of the application span). The _training loop span_ can include several operations (each represented as a span). Training loop span itself has several child spans. This structure allows us to reason about the relationship between different operations and attach metrics/attributes to each operation.

When a training job is integrated with this library, the above spans are created automatically. Moreover, for some of the spans, a set of predefined attributes (e.g., metrics, timing data, etc) are collected and reported to the telemetry backend.

Span Name             | Predefined Span Attributes
--------------------- | ----------------------------
TRAINING_LOOP         | TrainingLoopAttributes
CHECKPOINT_SAVE_SYNC  | CheckpointSaveSpanAttributes
CHECKPOINT_SAVE_ASYNC | CheckpointSaveSpanAttributes

Moreover, for each span, several events are trigerred (again automatically when a training job integrates with the library). The table below shows all the predefined events created for each span type and the predefined attributes collected and reported for each event: 

<!-- DO NOT CONVERT THIS TABLE TO A MARKDOWN TABLE. It has cells that arte merged vertically, which is not something markdown supports. -->
<table style="border-collapse: collapse; width: 100%; margin: 1rem 0;">
    <thead>
        <tr>
            <th style="border: 1px solid #000; padding: 8px; text-align: left;">Span Name</th>
            <th style="border: 1px solid #000; padding: 8px; text-align: left;">Event Name</th>
            <th style="border: 1px solid #000; padding: 8px; text-align: left;">Event Attributes Class Name</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;" rowspan="3">APPLICATION</td>
            <td style="border: 1px solid #000; padding: 8px;">SPAN_START</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;">ONE_LOGGER_INITIALIZATION</td>
            <td style="border: 1px solid #000; padding: 8px;">OneLoggerInitializationAttributes</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;">SPAN_STOP</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;" rowspan="3">TRAINING_LOOP</td>
            <td style="border: 1px solid #000; padding: 8px;">SPAN_START</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;">TRAINING_METRICS_UPDATE*</td>
            <td style="border: 1px solid #000; padding: 8px;">TrainingMetricsUpdateAttributes</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;">SPAN_STOP</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;">TRAINING_SINGLE_ITERATION</td>
            <td style="border: 1px solid #000; padding: 8px;">None</td>
            <td style="border: 1px solid #000; padding: 8px;">Instead of collecting metrics on each iteration, we use the TRAINING_MULTI_ITERATION_METRICS_UPDATE of the training loop event to control the amount of data sent to the backends</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;" rowspan="3">VALIDATION_LOOP</td>
            <td style="border: 1px solid #000; padding: 8px;">SPAN_START</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;">VALIDATION_METRICS_UPDATE*</td>
            <td style="border: 1px solid #000; padding: 8px;">ValidationMetricsUpdateAttributes</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;">SPAN_STOP</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;" rowspan="3">CHECKPOINT_SAVE_ASYNC or CHECKPOINT_SAVE_SYNC</td>
            <td style="border: 1px solid #000; padding: 8px;">SPAN_START</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;">SAVE_CHECKPOINT_SUCCESS</td>
            <td style="border: 1px solid #000; padding: 8px;">SaveCheckpointSuccessEventAttributes</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;">SYNC_CHECKPOINT_METRICS_UPDATE</td>
            <td style="border: 1px solid #000; padding: 8px;">SyncCheckpointMetricsUpdateAttributes</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td style="border: 1px solid #000; padding: 8px;">SPAN_STOP</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;" rowspan="3">TESTING_LOOP</td>
            <td style="border: 1px solid #000; padding: 8px;">SPAN_START</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;">TESTING_METRICS_UPDATE</td>
            <td style="border: 1px solid #000; padding: 8px;">TestingMetricsUpdateAttributes</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td style="border: 1px solid #000; padding: 8px;">SPAN_STOP</td>
            <td style="border: 1px solid #000; padding: 8px;">timestamp</td>
        </tr>
    </tbody>
</table>
> **_NOTE:_** 
"*" means that the parent span can have multiple instances of this span. For example, training loop has multiple spans of type TRAINING_SINGLE_ITERATION (one for each iteration of training).

<!-- DO NOT CONVERT THE ABOVE TABLE TO A MARKDOWN TABLE. It has cells that arte merged vertically, which is not something markdown supports. -->

 ## Integration with One Logger Training Telemetry

There are two ways to integrate with this library:

- If you are using a training framework that we support, you can simply use our glue code. For example, if you are using PyTorch lightning, all you need to do is to wrap your trainer class in our `OneLoggerPTLTrainer` class. In these cases, we use the callback mechanism of the underlying framework (Lightning in this case) to create spans, events, and collect attributes.

- If you have a custom training job or are using a framework that we don't support yet, you can simply use our telemetry APIs (context managers or callbacks) to tell the library what part of your code corresponds to major training events  (training loop, checkpoint loading or saving code, etc).
With this, the library will automatically create several spans, keeps tracks of relevant metrics, and exports them for you.

- For more advanced use cases, you can also call the core one logger API directly  (using `timed_span` for example).

Below, we will go into details for each of the above approaches.

### Integration through a supported framework

TODO: Add later

### Integration using context managers

Use the context managers defined in `src/one_logger/training_telemetry/api/context.py` to demarcate your main function, training loop, validation loop, etc.
See the example code at `src/one_logger/training_telemetry/docs/example.py`. Below is a simplified version of that code:

```python
@application() # < ---- telemetry context manager
def main() -> None:

    ....

    with training_loop(train_iterations_start=0): # < ---- telemetry context manager
        ...


        for epoch in range(num_epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                with training_iteration(): # < ---- telemetry context manager
                    ....

# Initialize the telemetry provider with a default configuration
config = TrainingTelemetryConfig(
    world_size_or_fn=5,
    is_log_throughput_enabled_or_fn=True,
    flops_per_sample_or_fn=100,
    global_batch_size_or_fn=32,
    log_every_n_train_iterations=10,
    application_name="test_app",
    perf_tag_or_fn="test_perf",
    session_tag_or_fn="test_session",
)

# configure the telemetry library and start the main() function
(TrainingTelemetryBuilder()
    .with_base_telemetry_config(config)
    .with_exporter(FileExporter(file_path=Path("training_telemetry.json")))
    .configure_provider())
main()
```



### Integration using callbacks

You can get training telemetry data by calling the calbacks defined in `src/one_logger/training_telemetry/api/callbacks.py`.

Here is a simplified example:

```python

def main() -> None:
    # configure the telemetry library and start the main() function
  (TrainingTelemetryBuilder()
      .with_base_telemetry_config(config)
      .with_exporter(FileExporter(file_path=Path("training_telemetry.json")))
      .configure_provider())

    on_app_start() # < ---- callback

    ....

        on_train_start(train_iterations_start=0) # < ---- callback
        ...
        for epoch in range(num_epochs):
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                on_training_single_iteration_start(): # < ---- callback
                    ....
                on_training_single_iteration_end(): # < ---- callback
    on_app_end() # < ---- callback
```

Note that you can combined this appraoch with calling the core onelogger API if you need to.
See [Integration using one logger core API](#integration-using-one-logger-core-api) for more info.


### Integration using one logger core API
One logger training telemetry library is built on top of the core one logger library. Therefore, you
have full access to the core API. Specifically,

- you can use the `Span` object created by the training context manager and then
  add attributes or create events.
- You can use the `timed_span` API to create your own spans.

```python
def main() -> None:
    # configure the telemetry library and start the main() function
    (TrainingTelemetryBuilder()
        .with_base_telemetry_config(config)
        .with_exporter(FileExporter(file_path=Path("training_telemetry.json")))
        # If you are creating custom spans, make sure you set the export_customization_mode and span_name_filter
        # such that such spans are exported.
        .with_export_customization(export_customization_mode=ExportCustomizationMode.xxxx, 
                                   span_name_filter=[...])
        .configure_provider())
    TrainingTelemetryProvider.instance().configure(
        config=config, 
        exporters=[],
)

    ....

    with training_loop(train_iterations_start=0) as span: # <---- access the span created by the context manager
        ...
        training_span.add_attribute("my_custom_attribute", "my_custom_value") # <--- Adding custom attributes

        with timed_span("my_custom_span", span_attributes=Attributes({"my_custom_attribute": "my_custom_value"})):
            # This code block is considered "my_custom_span"
            ....
            TrainingTelemetryProvider.instance().recorder.event(Event.create(...)) # <--- Firing a custom event
                            
```

### Comparison

The table below helps you choose the best integration approach based on your requirements.

| Using a supported <br/>training framework? | Need to define custom<br/>spans/events/attribs?* | Recommendation |
|---------------------|----------------------------------------------------------|----------------|
| Y                   | N                                                        | Use framework-level integration             |
| Y                   | Y                                                        | Use framework-level integration along with context managers for custom spans/events             |
| N                   | N                                                        | Use callbacks or context managers.<br/>The former has the advantage that it allows you to separate the telemetry code from training/model code<br/>(i.e. encapsulate all of telemetry code in callback functions).
| N                   | Y                                                        | Use context managers as callbacks only exist for predefined spans/events and<br/>the context managers (timed_span) are more readable and less error prone that calling the recorder API directly.



* _Custom spans/events/attributes_ refer to spans and events that are not predefined in the library (see `StandardTrainingJobSpanName` and `StandardTrainingJobEventName` enums and attributes defined in `attributes.py`
for a list of predefined spans/events/attributes). You may need to define custom spans/events/attributes if you want to collect telemetry data beyond what predefined spans/events/attributes collect.