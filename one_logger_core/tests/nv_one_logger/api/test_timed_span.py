# SPDX-License-Identifier: Apache-2.0
"""Tests for the timed_span functionality in the one_logger API."""

from typing import Generator
from unittest.mock import Mock

import pytest

from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.api.one_logger_provider import OneLoggerProvider
from nv_one_logger.api.recorder import Recorder
from nv_one_logger.api.timed_span import timed_span
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.exceptions import OneLoggerError
from nv_one_logger.core.span import SpanName


@pytest.fixture(autouse=True)
def configure_provider(config: OneLoggerConfig, mock_recorder: Recorder) -> None:
    """Fixture that configures the OneLoggerProvider."""
    # Reset the state of the singletons
    OneLoggerProvider.instance()._config = None  # type: ignore[reportPrivateUsage]
    OneLoggerProvider.instance()._recorder = None  # type: ignore[reportPrivateUsage]
    OneLoggerProvider.instance().configure(config, mock_recorder)


@pytest.fixture
def mock_recorder() -> Generator[Recorder, None, None]:
    """Fixture that sets up a mock recorder."""
    recorder = Mock(spec=Recorder)
    yield recorder

    recorder.reset_mock()


class TestTimedSpan:
    """Test suite for the timed_span functionality."""

    def test_timed_span_basic_usage(self, mock_recorder: Mock) -> None:
        """Test basic usage of timed_span as a context manager."""
        span_name: SpanName = "test_span"
        with timed_span(span_name) as span:
            pass

        mock_recorder.start.assert_called_once_with(span_name=span_name, span_attributes=None, start_event_attributes=None)
        mock_recorder.stop.assert_called_once_with(span)
        # Verify no other methods were called
        assert len(mock_recorder.method_calls) == 2

    def test_timed_span_with_attributes(self, mock_recorder: Mock) -> None:
        """Test timed_span with custom attributes."""
        span_name: SpanName = "test_span"
        span_attrs = Attributes({"span_key": "span_value"})
        start_attrs = Attributes({"start_key": "start_value"})

        with timed_span(span_name, span_attributes=span_attrs, start_event_attributes=start_attrs) as span:
            pass

        mock_recorder.start.assert_called_once_with(span_name=span_name, span_attributes=span_attrs, start_event_attributes=start_attrs)
        mock_recorder.stop.assert_called_once_with(span)
        # Verify no other methods were called
        assert len(mock_recorder.method_calls) == 2

    def test_nested_timed_span(self, mock_recorder: Mock) -> None:
        """Test nested timed_spana."""
        span_attrs = Attributes({"span_key": "span_value"})
        start_attrs = Attributes({"start_key": "start_value"})

        with timed_span("outer_span", span_attributes=span_attrs, start_event_attributes=start_attrs):
            with timed_span("inner_span"):
                pass

        # Verify start calls
        assert mock_recorder.start.call_count == 2
        mock_recorder.start.assert_any_call(span_name="outer_span", span_attributes=span_attrs, start_event_attributes=start_attrs)
        mock_recorder.start.assert_any_call(span_name="inner_span", span_attributes=None, start_event_attributes=None)

        # Verify stop calls
        assert mock_recorder.stop.call_count == 2
        # Verify no other methods were called
        assert len(mock_recorder.method_calls) == 4

    def test_timed_span_with_error(self, mock_recorder: Mock) -> None:
        """Test timed_span when an error occurs within the context."""
        span_name: SpanName = "test_span"
        test_error = ValueError("Test error")
        the_span = None
        with pytest.raises(ValueError) as e:
            with timed_span(span_name) as span:
                the_span = span
                raise test_error

        assert e.value == test_error
        mock_recorder.start.assert_called_once_with(span_name=span_name, span_attributes=None, start_event_attributes=None)
        mock_recorder.error.assert_called_once_with(the_span, f"Error in {span_name}:", test_error)
        mock_recorder.stop.assert_called_once_with(the_span)
        # Verify no other methods were called
        assert len(mock_recorder.method_calls) == 3

    def test_set_recorder_twice(self) -> None:
        """Verify that set_recorder can only be called once."""
        with pytest.raises(OneLoggerError, match="OneLoggerProvider already configured!"):
            OneLoggerProvider.instance().configure(Mock(spec=OneLoggerConfig), Mock(spec=Recorder))
