# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from typing import Generator, Optional

from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.api.recorder import Recorder
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.core.span import Span, SpanName

_logger = get_logger(__name__)


def get_recorder() -> Recorder:
    """Return the singleton recorder.

    This is a simply a shortcut for getting the recorder used by the timed_span context manager.
    """
    return OneLoggerProvider.instance().recorder  # type: ignore[no-any-return]


@contextmanager
def timed_span(
    name: SpanName,
    span_attributes: Optional[Attributes] = None,
    start_event_attributes: Optional[Attributes] = None,
) -> Generator[Span, None, None]:
    """
        Context manager for recording a span.

        Note that OneLoggerProvider.instance().configure() must be called once per process (e.g., at app start up) before using the timed_span context manager.
        See the "How to use One Logger" section of the README.md file for more information.

        Args:
            name: The name of the span.
            span_attributes: Optional attributes to add to the span.
            start_event_attributes: Optional attributes to add to the start event.

        Yields:
            the span


        Usage:
        On app start up:
        ```python
            from nv_one_logger.api.timed_span import get_recorder, timed_span

            def main():
                # One-time initialization of the library
                recorder = ... # You can use a factory that takes some config parameters and builds a recorder.
                            # Or in simple use cases, just use recorder = DefaultRecorder(exporters=[...])
                OneLoggerProvider.instance().configure(recorder)
        ```

        In other parts of the application:
        ```python
            from nv_one_logger.api.timed_span import get_recorder, timed_span

            def foo():
                with timed_span(name="foo", start_event_attributes=Attributes({...})):
                    # business logic for operation foo()
        ```

        or for more advanced use cases, you can access the recorder directly for recording events and errors:
        ```python

            def bar():
                with timed_span(name="application", span_attributes=Attributes({"app_tag": "some tag"})) as span:
                    # some business logic

                    # You can record events of interests within a timed span.
                    get_recorder().event(span, Event.create(...))

                    # Or record errors
                    if(...):
                        get_recorder().error(span, "Error in bar", ...)

                    # some more business logic

    ```

    """
    recorder = get_recorder()
    assert_that(recorder, "One Logger is not configured! Use OneLoggerProvider.instance().configure() to configure One Logger.")
    span = recorder.start(span_name=name, span_attributes=span_attributes, start_event_attributes=start_event_attributes)

    try:
        yield span
    except Exception as e:
        recorder.error(span, f"Error in {name}:", e)
        raise e
    finally:
        recorder.stop(span)
