# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MetricSummarizer class."""

from nv_one_logger.core.internal.metric_summarizer import MetricSummarizer


def test_metric_summarizer_initial_state() -> None:
    """Test that a new MetricSummarizer instance has the correct initial state."""
    summarizer = MetricSummarizer[int]()
    assert summarizer.latest_value is None
    assert summarizer.min_value is None
    assert summarizer.max_value is None
    assert summarizer.avg_value is None
    assert summarizer.total_value is None
    assert summarizer.count == 0


def test_metric_summarizer_add_single_value() -> None:
    """Test adding a single value to the MetricSummarizer."""
    summarizer = MetricSummarizer[int]()
    summarizer.add_value(42)

    assert summarizer.latest_value == 42
    assert summarizer.min_value == 42
    assert summarizer.max_value == 42
    assert summarizer.avg_value == 42.0
    assert summarizer.total_value == 42
    assert summarizer.count == 1


def test_metric_summarizer_add_multiple_values() -> None:
    """Test adding multiple values to the MetricSummarizer."""
    summarizer = MetricSummarizer[int]()
    values = [10, 20, 30, 40, 50]

    for value in values:
        summarizer.add_value(value)

    assert summarizer.latest_value == 50
    assert summarizer.min_value == 10
    assert summarizer.max_value == 50
    assert summarizer.avg_value == 30.0
    assert summarizer.total_value == 150
    assert summarizer.count == 5


def test_metric_summarizer_with_float_values() -> None:
    """Test MetricSummarizer with float values."""
    summarizer = MetricSummarizer[float]()
    values = [1.5, 2.5, 3.5]

    for value in values:
        summarizer.add_value(value)

    assert summarizer.latest_value == 3.5
    assert summarizer.min_value == 1.5
    assert summarizer.max_value == 3.5
    assert summarizer.avg_value == 2.5
    assert summarizer.total_value == 7.5
    assert summarizer.count == 3


def test_metric_summarizer_equality() -> None:
    """Test equality comparison between MetricSummarizer instances."""
    summarizer1 = MetricSummarizer[int]()
    summarizer2 = MetricSummarizer[int]()

    # Test empty summarizers
    assert summarizer1 == summarizer2

    # Add same values to both
    values = [1, 2, 3]
    for value in values:
        summarizer1.add_value(value)
        summarizer2.add_value(value)

    assert summarizer1 == summarizer2

    # Add different values
    summarizer1.add_value(4)
    assert summarizer1 != summarizer2


def test_metric_summarizer_hash() -> None:
    """Test hash functionality of MetricSummarizer."""
    summarizer1 = MetricSummarizer[int]()
    summarizer2 = MetricSummarizer[int]()

    # Test empty summarizers
    assert hash(summarizer1) == hash(summarizer2)

    # Add same values to both
    values = [1, 2, 3]
    for value in values:
        summarizer1.add_value(value)
        summarizer2.add_value(value)

    assert hash(summarizer1) == hash(summarizer2)

    # Add different values
    summarizer1.add_value(4)
    assert hash(summarizer1) != hash(summarizer2)
