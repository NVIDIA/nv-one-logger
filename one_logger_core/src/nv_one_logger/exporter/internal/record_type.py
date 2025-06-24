# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class RecordType(Enum):
    """The type of record to be used by backends that log to a file, e.g. FileBackend and LoggerBackend."""

    # The start of a span.
    START = "start"
    # The end of a span.
    STOP = "stop"
    # A complete span, that is the start and stop of a span are combined into a single record for efficiency.
    COMPLETE = "complete"
    # A single event that can occur anytime in a span, e.g. a training iteration in a training loop.
    EVENT = "event"
    # An error is a special event that is used to record an exception or error encountered in the application.
    APPLICATION_ERROR = "application_error"
    # An error is a special event that is used to record an exception or error encountered in one logger code.
    TELEMETRY_DATA_ERROR = "telemetry_data_error"
