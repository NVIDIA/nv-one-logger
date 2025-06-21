# one_logger_core

## Summary

This Python project contains the API and libraries for One Logger: A library for collecting telemetry information from jobs.

The following are the dependency rules for various packages (which we will enforce by creating separate python projects in the end state):

- core: Classes representing core concepts such as span, event, attributes. All other packages can depend on this package but this package must not depend on any one_logger or telemetry package. Moreover, we must try to minimize the third-party dependencies of this package so that it remains lightweight and easy to adopt by internal and external users (any new dependency can conflict with existing dependencies of the app that is being instrumented).

- exporter: This package will contain code for various supported exporters. We have a few simple exporters in one_logger_core project. Vendor-specific exporters (OTEL, Kafka, etc) can be added to extend the system. However, each exporter must be added in a separate Python project so that we don't pull in extra dependencies in the `one_logger_cor`e` project.

- api: This package contains the API to record spans and events.

The application is expected to depend on `one_logger_core` project and optionally on any project that contains vendor-specific exporters.

## Concepts

We have defined our abstractions based on open telemetry concepts.

- **Span**: A span represents a unit of work or operation. It can contain events, and can have attributes associated with it. Spans are used to track the execution of operations and their relationships in a distributed system. A span can have a set of events as well as attributes associated with it. In our library, we ensure that a start event is created when the span is created and an end event when the span is stopped. [more info](https://opentelemetry.io/docs/concepts/signals/traces/#spans).
- **Event**: A Span Event represents a meaningful, singular point in time during the Span's duration. Each event has a name, timestamp, and a set of attributes. [more info](https://opentelemetry.io/docs/concepts/signals/traces/#span-events).
- **Attribute**: A property of a span or event.
- **Exporter**: An exporter sends the data from the instrumented application to an observability backend. This can be Kafka, an experiment management system such as Weights and Biases, or a `collector pipeline`. The exporter is responsible to format and serialize data for a particular backend type. For production environments, Open Telemetry recommends exporting the data to a collector pipeline, which then can process and further export the data to the final destination(s). Using a collector pipeline allows the local exporter to stay vendor agnostic, simply send the data to a collector pipeline (e.g., a cluster-level OTEL receiver) and let the collector deal with communication with vendor-specific solutions and backends (serialization for the vendor, retries, filtering sensitive data, and even sending data to multiple backends). Note that this is different from OpenTelemetry's exporter

- **Recorder**: A Recorder makes using all of the above easier. The application can call the Recorder API to start/stop spans, add events, or report errors. The Recorder is in charge of creating the Span/Event objects and then using one or more exporters to send the spans, events, and errors to one or more backends. The recorder decides which exporters to use and when to call them. Here are a few examples of what can be done in a Recorder:

  - Filter (not export) certain events (based on attribute values or verbosity).
  - Add new attributes to a span or events or even create new events. For example, if you have a long-running span for model training in which multiple "save checkpoint" events are emitted, and you want to keep track of the avg, max, and min save times across all the checkpoints, the recorder can keep track of all "save checkpoint" events so far and maintain avg, max, and min values of the "duration" attribute of those events and then emit a "check point stats update" event periodically.

## How to use One Logger

There are 3 options for colleting telemetry information from applications (including long-running jobs such as ML training):

- Using higher-level domain-specific libraries built on top of One Logger (e.g., one_logger_training_telemetry library). These libraries expose high-level APIs that allow the user to capture well-known operations and events in that domain (e.g., reporting the start/completion of checkpointing in training). If your application is in the domain of one of these high-level libraries, this approach is preferred.

- Using an implementation of the `one_logger.api.Recorder` interface (e.g., `one_logger.recorder.DefaultRecorder`) directly or via the `one_logger.api.timed_span` context manager. Here are a few examples:

```python
from nv_one_logger.api.timed_span import configure_one_logger, get_recorder, timed_span

def main():
    # One-time initialization of the library
    config = OneLoggerConfig(....) # You can use a factory that takes a json or other representation of the configs and creates a OneLoggerConfig object.
    recorder = ... # You can use a factory that takes some config parameters and builds a recorder. 
                   # Or in simple use cases, just use recorder = DefaultRecorder(exporters=[...])
    OneLoggerProvider.instance().configure(config, recorder)

    # A span corresponding to the entire execution of the application. All the work done within the
    # "with timed_span" block will be considered part of a new span.
    with timed_span(name="application", span_attributes=Attributes({"app_tag": "some tag"})):
        # some business logic
        foo()
        # some more business logic

def foo():
    # Another span corresponding to operation Foo
    with timed_span(name="foo", start_event_attributes=Attributes({...})) as span:
        # business logic for operation Foo

        # You can record events of interests within a timed span.
        get_recorder().event(span, Event.create(...))

        # Or record errors
        if(...):
            get_recorder().error(span, ...)
```

Or you can simply use the `Recorder` interface directly:

```python
def main():

    # One-time initialization at app start up.

    config = OneLoggerConfig(....) # You can use a factory that takes a json or other representation of the configs and creates a OneLoggerConfig object.
    recorder = ... # You can use a factory that takes some config parameters and builds a recorder. 
                   # Or in simple use cases, just use recorder = DefaultRecorder(exporters=[...])
    OneLoggerProvider.instance().configure(config, recorder)

def some_func():
    # ....

    recorder.event(...)

    # ....

    recorder.error(...)

    # ....

    recorder.stop(span)
```

- You can also bypass the `Recorder` interface and `timed_span` and directly instantiate one of more `Exporter`s and use them to export spans and events you have created using the classes under `one_logger.core`. Try to avoid this approach unless really necessary. Using a `Recorder` is preferred as it reduces the chance of making mistakes.

## Design Considerations

While we could have used Python classes defined in the OpenTelemetry API (such as Span, Attribute, etc.) instead of creating our own classes, this would force any application using our core library to depend on both the OpenTelemetry API and SDK (the latter is needed in the bootstrap code that creates a provider object for creation of spans). This dependency requirement could potentially conflict with the application's existing dependencies or create unnecessary constraints (e.g., when an application already depends on a different version of OpenTelemetry SDK for its own instrumentation needs. A similar problem can happen with conflicting transitive dependencies).

This design decision means applications can use our core library without worrying about OpenTelemetry dependency conflicts, while still benefiting from OpenTelemetry-compatible instrumentation patterns. Users who would like to use OpenTelemetry collectors as their backend, can easily map the data to OpenTelemetry Python classes in the "backend" class.

## Configuration

One Logger is meant to add instrumentation to applications to track performance of the application. One of the important usages of One Logger is to identify significant performance changes that are not expected. That is, we run an application once and collect baseline performance data. Then, every time we run the application again, we can compare the performance data against the `baseline`. These comparisons are only useful if we can differentiate between cases that changes in the performance are expected and those that are not.

Another use case is to measure the impact of a change in code, job configuration, or execution environment on the performance of an application.

To be able to do the above easily, one logger supports tagging each run with some extra metadata to allow meaningful comparisons of runs. `perf_tag` and `session_tag` parameters are created for this reason. The user of the library can set the appropriate values as part of configuring one logger. Those values will be exported alongside the telemetry data to a telemetry backend. When interpretting, analyzing, or aggregating telemetry data, these tags provide extra context about the job to

- flag anomalies in the performance (and unexpected performance degradation).
- track progress of the application even if the progress is made by different jobs across different machines or clusters.
- track the performance of the application over time and correlate performance changes with code changes.
- and more.

Below, we will explain the semantics of `perf_tag` and `session_tag` and the relationship between them. We expect the user to ensure these values are set correctly for each execution of their job.

`perf_tag`: used to identify jobs whose performance is expected to be comparable. This means jobs with the same perf_tag must be performing similar tasks and are using the same code, config, and resources (or only differ in ways that are not expected to impact the performance).

`session_tag`: used to identify jobs that together contribute to the same task. This means the jobs are "logically" part of a single larger job (e.g., a hypothetical long running job that is split into multiple jobs due to resource constraints or resuming after a failure).

Let's use a few examples to illustrate the usage of these knobs. We use a model training application to illustrate the usage of these knobs but the same concepts apply to other application types.

Imagine we have a model training application. A user downloads a snapshot of the code of this application (say a git branch at a certain commit) and runs the application on some hardware with some configuration (number of GPUs, batch size, etc). Let's assume the user needs to run 1000 iterations of training to complete the task (train a model with acceptable accuracy). Now let's go through a few scenarios:

Scenario 1: User runs the job. It completes without a problem and fully trains the model. A week later, the user changes the model architecture significantly and then runs the job again. Due to the fundamental change in the job code, the two runs are not expected to have similar performance characteristics . In this case, the user should assign different values to "session_tag" across the two jobs because the two runs are independent from each other (they are independent training sessions each training the model from scratch to completion). Moreover, the user must assign different values to "perf_tag" because the two runs are not expected to have similar performance characteristics due to the changes in model code.

Scenario 2: Simialr to scenario 1 except that for the second execution of the job, instead of changing the model architecture, the user allocates more resources to the job (the code remains the same). In this case, the user should assign different values to "session_tag" across the two jobs because the two runs are independent from each other (they are independent training sessions each training the model from scratch to completion). Moreover, the user must assign different values to "perf_tag" because the two runs are not expected to have similar performance characteristics due to the changes in resources.

Scenario 3: The user runs the job to completion. The next day, there is an OS upgrade performed on the cluster to apply a security patch, which in theory should not impact the performance of the jobs on that cluster. The user runs the job again without changing the code or config. In this scenario: since the two runs are independently training the model from scratch to completion (in other words, are not part of the same training session), each run should get a different value for session_tag. However, since the two executions used the same code, config, and execution environment, they must have the same perf_tag.

Scenario 4: The user runs the job but it fails due to an issue at iteration 100 (e.g., due a hardware issue, a scheduling constraint causing the job to be evicted, or a small bug in the model code). The user fixes the issues and runs the job again. In this scenario, the fix is not expected to significantly change the performance characteristics of the job. Since the user is using training chekcpoints, the second run will resume training from iteration 90 when the last checkpoint was saved. Once the second run completes, we have a fully trained model. In this scenario, the two runs are indeed logically part of the same task (same training session) and are expected to have the same performance characteristics as we didn't change the code, hardware, or configs in any way that we expect to impact the performance. So the two runs should have the same perf_tag and session_tag values.

Scenario 5: The user runs the job but it fails at iteration 100 due to an issue in model code. To fix the issue, the user makes a change to the model code that, in addition to fixing the bug, significantly speeds up training (e.g, an unnecessary loop is removed). The user runs the job again and due to checkpointing, the second run starts from iteration 90 when the last checkpoint was saved. In this scenario, the two runs are indeed logically part of the same task (training the model from scratch to completion) but they are not expected to have the same performance characteristics due to the above-mentioned change in code. In this case, the two runs should have the same session_tag but different vaues for perf_tag.

In summary, when configuring one logger for a particular application,

- Change the value of `perf_tag`, whenever a change is made that is expected to change the performance charactristics of the application (changes in code, config/resources, or execution environment).

- Use a unique value for `session_tag` for each single logical execution of your application (if a single logical execution is spread across multiple physical jobs due to interruptions in execution, all those jobs must have the same session_tag).

## Dealing with Telemetry Failures

Like any other piece of software, the one logger library may encounter failures due to misconfiguration, incorrect usage, connection issues with the telemetry backends, or internal bugs in the library. For the rest of this section, we collectively refer to these issues as **telemetry errors**. The library provides several mechanisms to handle telemetry errors correctly:

- Users of the library can choose how telemetry errors are handled using `config.error_handling_strategy`. This enum allows users to treat telemetry as a critical part of the application (potentially letting telemetry errors cause a crash) or as a non-critical component (gracefully handling errors). Please see `OneLoggerErrorHandlingStrategy` enum for more information.

- Regardless of the error handling strategy, If the library encounters a telemetry error, the data exported to the telemetry backends is not guaranteed to be correct anymore. In such cases, the library calls the `export_telemetry_data_error` method of all the exporters. The implementation of this method in each type of Exporter is responsible to send a signal to the corresponding telemetry backend that informs the backend the colelcted data is not reliable. Make sure you familiarize yourself with how the exporters that you use send this signal and have the backend and the any analytics code you write for the backend data use that signal to exclude the data from analytics.

- Note that if you choose to use `DefaultRecorder`, there is another mechanism to help with errors. DefaultRecorder monitors the errors encountered while exporting data to each exporter (note that any recorder can be configured to export to multiple exporters). If a telemetry error is related to exporting to a certain exporter (as opposed to a general issue such as inconsistent telemetry state or misconfigured library). `DefaultRecorder` calls `export_telemetry_data_error` only on that exporter (as the data exported to the other exporters is not affected by that issue). If the exporter continues to fail frequently, `DefaultRecorder` disables that exporter.

In summary:

- Choose your desired error handling stratgy via `config.error_handling_strategy`.
- For each backend, the corresponding Exporter implementation sends some info about missing or corrupted data to that backend. When using the telemetry data stored on your telemetry backend, pay attention to reports about data issues.
