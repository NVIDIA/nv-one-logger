# SPDX-License-Identifier: Apache-2.0
"""Provider for the OneLogger singleton instance.

This module provides a singleton class for managing the global OneLogger instance.
"""

from typing import Optional

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.api.recorder import Recorder
from nv_one_logger.core.exceptions import assert_that
from nv_one_logger.core.internal.singleton import SingletonMeta


class OneLoggerProvider(metaclass=SingletonMeta["OneLoggerProvider"]):
    """A singleton class provider for OneLogger.

    This singleton provides a global point of entry for the OneLogger library.

    Why do we need this singleton?
    - Implementations of Recorder are often stateful. So an application needs to create a single Recorder and use it everywhere.
    - The code that uses this library for instrumentation needs to get access to the config of the library and the recorder. Without
      a singleton, the application would need to be changed to plumb the instance of Recorder and the config everywhere. This makes
      the ligrary harder to adopt.


    This singleton needs to be configured once per process as follows on application start up:
    ```python
        config = OneLoggerConfig(....) # You can use a factory that takes a json or other representation of the configs and creates a OneLoggerConfig object.
        recorder = ... # You can use a factory that takes some config parameters and builds a recorder.
                   # Or in simple use cases, just use recorder = DefaultRecorder(exporters=[...])
        OneLoggerProvider.instance().configure(config, recorder)
    ```

    Once configured, the application can use the singleton:
    ```python
        # In other parts of the application
        with OneLoggerProvider.instance().recorder.start("my_span"):
            ... # code here will be considered part of the span and will be timed/recorded.

        or use timed_span context manager.
    ```
    """

    _recorder: Optional[Recorder] = None
    _config: Optional[OneLoggerConfig] = None

    """
    If set to True, the logging will be forced to be disabled effectively disabling onelogger library.
    This flag is useful when the user configures onelogger to enable logging but later on, we decide to
    disable logging (e.g., the library itself decides to disable logging due to encountering errors during export).
    """
    _logging_force_disabled: bool = False

    def configure(self, config: OneLoggerConfig, recorder: Recorder) -> None:
        """
        Set the recorder for the singleton.

        Args:
            recorder: The recorder to set.
        """
        assert_that(config, "config cannot be None.")
        assert_that(
            (not self._recorder or self._recorder is recorder) and (not self._config or self._config is config),
            "OneLoggerProvider already configured! You must call configure() once and only once.",
        )
        self._config = config
        self._recorder = recorder

    @property
    def recorder(self) -> Recorder:
        """Return the recorder."""
        assert_that(self._recorder, "You need to call OneLoggerProvider.instance().configure() once in your application before accessing the recorder.")
        return self._recorder  # type: ignore[return-value]

    @property
    def config(self) -> OneLoggerConfig:
        """Return the config."""
        assert_that(self._config, "You need to call OneLoggerProvider.instance().configure() once in your application before accessing the config.")
        return self._config  # type: ignore[return-value]

    @property
    def one_logger_ready(self) -> bool:
        """Check if the one_logger is ready to be used."""
        return self._recorder is not None and self._config is not None

    @property
    def one_logger_enabled(self) -> bool:
        """Check if the one_logger is ready to be used and logging is not forced to be disabled."""
        return self.one_logger_ready and not self._logging_force_disabled and self.config.enable_one_logger

    def force_disable_logging(self) -> None:
        """Force logging to be disabled effectively disabling onelogger library."""
        self._logging_force_disabled = True
