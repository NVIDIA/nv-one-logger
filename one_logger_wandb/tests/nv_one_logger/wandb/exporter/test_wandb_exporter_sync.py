# SPDX-License-Identifier: Apache-2.0
"""Test suite for the WandBExporter class."""

from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import (
    ErrorEvent,
    Event,
    StandardEventAttributeName,
    StandardEventName,
    TelemetryDataError,
)
from nv_one_logger.core.span import Span, StandardSpanAttributeName
from nv_one_logger.core.time import TracingTimestamp
from wandb.sdk.wandb_run import Run

from nv_one_logger.wandb.exporter.wandb_exporter import (
    Config,
    FlatMetricNamingStrategy,
    HierarchicalMetricNamingStrategy,
    MetricNamingStrategy,
    WandBExporterSync,
)
from tests.nv_one_logger.wandb.exporter.helpers import advance_time


class TestWandBExporterSync:
    """Test suite for the WandBExporterSync class."""

    @pytest.fixture(scope="function")
    def exporter(self, request: pytest.FixtureRequest) -> Generator[WandBExporterSync, None, None]:
        """Create and initialize a WandBExporterSync instance.

        Args:
            request: The pytest fixture request object containing the parametrized metric naming strategy.

        Yields:
            An initialized WandBExporterSync instance.
        """
        metric_naming_strategy: MetricNamingStrategy = request.param
        with patch("wandb.init") as mock_init, patch("wandb.login") as mock_login:
            self.mock_wandb_run = MagicMock(spec=Run)
            run_finish_mock = MagicMock()
            self.mock_wandb_run.finish = run_finish_mock
            mock_init.return_value = self.mock_wandb_run
            mock_login.return_value = None

            exporter = WandBExporterSync(Config(entity="test_entity", project="test_project"), metric_naming_strategy=metric_naming_strategy)
            assert not exporter.ready
            exporter.initialize()
            assert exporter.ready
            yield exporter

            # clean up
            exporter.close()
            assert exporter.closed
            run_finish_mock.assert_called_once()

    def _get_logged_metrics(self, exporter: WandBExporterSync) -> List[Dict[str, Any]]:
        """Get the metrics logged by the exporter."""
        return [call_arg.kwargs["data"] for call_arg in exporter._run.log.call_args_list]  # type: ignore[reportPrivateUsage]

    @pytest.mark.parametrize(
        "exporter, expected_logged_metrics",
        [
            (
                FlatMetricNamingStrategy(),
                [
                    # span start
                    {f"{StandardEventAttributeName.TIMESTAMP_MSEC}": 500_000},
                    # span stop
                    {
                        f"{StandardSpanAttributeName.DURATION_MSEC}": 5310,
                        f"{StandardEventAttributeName.TIMESTAMP_MSEC}": 505_310,
                    },
                ],
            ),
            (
                HierarchicalMetricNamingStrategy(),
                [
                    # span start
                    {f"test_span1.span_start.{StandardEventAttributeName.TIMESTAMP_MSEC}": 500_000},
                    # span stop
                    {
                        "test_span1.duration_msec": 5310,
                        f"test_span1.span_stop.{StandardEventAttributeName.TIMESTAMP_MSEC}": 505_310,
                    },
                ],
            ),
        ],
        indirect=["exporter"],
    )
    def test_export_span_without_attributes(self, exporter: WandBExporterSync, expected_logged_metrics: List[Dict[str, Any]]) -> None:
        """Test that a span without attributes is correctly exported to WandB."""
        # Create a test span
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=500, perf_counter=100, validate_timestamp=False)
        test_span = Span.create(
            name="test_span1",
            start_time=start_time,
        )
        exporter.export_start(test_span)

        stop_time = advance_time(start_time, 5.31)
        test_span.stop(stop_time=stop_time)
        exporter.export_stop(test_span)

        # Verify that the span was logged with correct parameters
        assert self._get_logged_metrics(exporter) == expected_logged_metrics

    @pytest.mark.parametrize(
        "exporter, expected_logged_metrics",
        [
            (
                FlatMetricNamingStrategy(),
                [
                    # span start
                    {
                        "span_attrib_key": "span_attrib_val",
                        "start_event_attr_key": "start_event_attr_val",
                        f"{StandardEventAttributeName.TIMESTAMP_MSEC}": 500_000,
                    },
                    # span stop
                    {
                        f"{StandardSpanAttributeName.DURATION_MSEC}": 5310,
                        "stop_event_attr_key": "stop_event_attr_val",
                        f"{StandardEventAttributeName.TIMESTAMP_MSEC}": 505_310,
                    },
                ],
            ),
            (
                HierarchicalMetricNamingStrategy(),
                [
                    # span start
                    {
                        "test_span2.span_attrib_key": "span_attrib_val",
                        "test_span2.span_start.start_event_attr_key": "start_event_attr_val",
                        f"test_span2.span_start.{StandardEventAttributeName.TIMESTAMP_MSEC}": 500_000,
                    },
                    # span stop
                    {
                        "test_span2.duration_msec": 5310,
                        "test_span2.span_stop.stop_event_attr_key": "stop_event_attr_val",
                        f"test_span2.span_stop.{StandardEventAttributeName.TIMESTAMP_MSEC}": 505_310,
                    },
                ],
            ),
        ],
        indirect=["exporter"],
    )
    def test_export_span_with_attributes(self, exporter: WandBExporterSync, expected_logged_metrics: List[Dict[str, Any]]) -> None:
        """Test that a span with attributes is correctly exported to WandB."""
        # Create a test span with attributes
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=500, perf_counter=100, validate_timestamp=False)
        test_span = Span.create(
            name="test_span2",
            span_attributes=Attributes(
                {
                    "span_attrib_key": "span_attrib_val",
                }
            ),
            start_event_attributes=Attributes(
                {
                    StandardEventAttributeName.TIMESTAMP_MSEC: start_time.milliseconds_since_epoch,
                    "start_event_attr_key": "start_event_attr_val",
                }
            ),
            start_time=start_time,
        )
        exporter.export_start(test_span)

        stop_time = advance_time(start_time, 5.31)
        test_span.stop(
            stop_time=stop_time,
            stop_event_attributes=Attributes(
                {
                    StandardEventAttributeName.TIMESTAMP_MSEC: stop_time.milliseconds_since_epoch,
                    "stop_event_attr_key": "stop_event_attr_val",
                }
            ),
        )
        exporter.export_stop(test_span)

        # Verify that the span was logged with correct parameters
        assert self._get_logged_metrics(exporter) == expected_logged_metrics

    @pytest.mark.parametrize(
        "exporter, expected_logged_metrics",
        [
            (
                FlatMetricNamingStrategy(),
                [
                    # span start
                    {"span_attrib_key": "span_attrib_val", StandardEventAttributeName.TIMESTAMP_MSEC: 500_000},
                    # No metric are emitted for errors.
                ],
            ),
            (
                HierarchicalMetricNamingStrategy(),
                [
                    # span start
                    {"test_span1.span_attrib_key": "span_attrib_val", f"test_span1.span_start.{StandardEventAttributeName.TIMESTAMP_MSEC}": 500_000},
                    # No metric are emitted for errors.
                ],
            ),
        ],
        indirect=["exporter"],
    )
    def test_export_error(self, exporter: WandBExporterSync, expected_logged_metrics: List[Dict[str, Any]]) -> None:
        """Test that error events are correctly exported to WandB."""
        # Create a test span
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=500, perf_counter=100, validate_timestamp=False)
        test_span = Span.create(
            name="test_span1",
            span_attributes=Attributes(
                {
                    "span_attrib_key": "span_attrib_val",
                }
            ),
            start_time=start_time,
        )
        exporter.export_start(test_span)

        # Create an error event
        error_time = advance_time(start_time, 2.5)
        error_event = ErrorEvent(
            timestamp=error_time,
            error_message="Test error message",
            exception_type="RuntimeError",
            exception_message="test exception",
            exception_traceback="test exception traceback",
        )
        exporter.export_error(error_event, test_span)

        # Verify that the error doesn't get exported.
        assert self._get_logged_metrics(exporter) == expected_logged_metrics

    @pytest.mark.parametrize(
        "exporter, expected_logged_metrics",
        [
            (
                FlatMetricNamingStrategy(),
                [
                    # span start
                    {StandardEventAttributeName.TIMESTAMP_MSEC: 600_000},
                    # event.
                    {"ev1_attr1": "ev1_val1", StandardEventAttributeName.TIMESTAMP_MSEC: 602_500},
                ],
            ),
            (
                HierarchicalMetricNamingStrategy(),
                [
                    # span start
                    {"test_span1.span_start.timestamp_msec": 600_000},
                    # event.
                    {"test_span1.event1.ev1_attr1": "ev1_val1", f"test_span1.event1.{StandardEventAttributeName.TIMESTAMP_MSEC}": 602_500},
                ],
            ),
        ],
        indirect=["exporter"],
    )
    def test_export_event(self, exporter: WandBExporterSync, expected_logged_metrics: List[Dict[str, Any]]) -> None:
        """Test that events are correctly exported to WandB."""
        start_time = TracingTimestamp.for_timestamp(timestamp_sec=600, perf_counter=100, validate_timestamp=False)
        test_span = Span.create(
            name="test_span1",
            start_time=start_time,
        )
        exporter.export_start(test_span)

        # Create an event
        event1_timestamp = advance_time(start_time, 2.5)
        event1 = Event.create(
            name="event1",
            attributes=Attributes(
                {
                    "ev1_attr1": "ev1_val1",
                    StandardEventAttributeName.TIMESTAMP_MSEC: event1_timestamp.milliseconds_since_epoch,
                }
            ),
            timestamp=event1_timestamp,
        )
        exporter.export_event(event1, test_span)

        # Verify that the error doesn't get exported.
        assert self._get_logged_metrics(exporter) == expected_logged_metrics

    @pytest.mark.parametrize(
        "exporter",
        [
            FlatMetricNamingStrategy(),
            HierarchicalMetricNamingStrategy(),
        ],
        indirect=["exporter"],
    )
    @pytest.mark.parametrize(
        "error_type",
        list(TelemetryDataError.ErrorType),
    )
    def test_export_telemetry_data_error(self, exporter: WandBExporterSync, error_type: TelemetryDataError.ErrorType) -> None:
        """Test that telemetry data errors are correctly exported to WandB."""
        timestamp = TracingTimestamp.for_timestamp(timestamp_sec=200, perf_counter=100, validate_timestamp=False)
        telemetry_data_error = TelemetryDataError.create(
            error_type=error_type,
            error_message="Test telemetry data error message",
        )
        telemetry_data_error._timestamp = timestamp  # type: ignore[reportPrivateUsage]
        exporter.export_telemetry_data_error(telemetry_data_error)
        assert self._get_logged_metrics(exporter) == [{StandardEventName.TELEMETRY_DATA_ERROR.value: error_type.name}]
