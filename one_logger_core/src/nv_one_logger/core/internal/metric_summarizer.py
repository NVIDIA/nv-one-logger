# SPDX-License-Identifier: Apache-2.0

"""A class that maintains some summary statistics for a numeric metric."""


from __future__ import annotations

from typing import Generic, Optional, TypeVar

T = TypeVar("T", int, float)


class MetricSummarizer(Generic[T]):
    """
    A class that maintains some summary statistics for a numeric metric.

    Type Parameters:
        T: The type of the metric value, must be either int or float
    """

    latest_value: Optional[T] = None
    min_value: Optional[T] = None
    max_value: Optional[T] = None
    avg_value: Optional[float] = None
    total_value: Optional[T] = None
    count: int = 0

    def add_value(self, value: T) -> None:
        """Add a value to the metric summarizer."""
        self.count += 1
        self.latest_value = value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)
        self.total_value = value if self.total_value is None else self.total_value + value
        self.avg_value = value if self.avg_value is None else float(self.total_value) / self.count

    def __hash__(self) -> int:
        """
        Compute the hash value of this MetricSummarizer instance.

        Returns:
            int: A hash value based on the current state of the summarizer
        """
        return hash((self.latest_value, self.min_value, self.max_value, self.avg_value, self.total_value, self.count))

    def __eq__(self, other: object) -> bool:
        """
        Compare this MetricSummarizer instance with another object for equality.

        Args:
            other: The object to compare with

        Returns:
            bool: True if the other object is a MetricSummarizer with the same state,
                 False otherwise
        """
        if not isinstance(other, MetricSummarizer):
            return False
        other_summarizer: MetricSummarizer[T] = other
        return (
            self.latest_value == other_summarizer.latest_value
            and self.min_value == other_summarizer.min_value
            and self.max_value == other_summarizer.max_value
            and self.avg_value == other_summarizer.avg_value
            and self.total_value == other_summarizer.total_value
            and self.count == other_summarizer.count
        )
