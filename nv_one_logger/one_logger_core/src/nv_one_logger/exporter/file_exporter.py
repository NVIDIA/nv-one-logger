# SPDX-License-Identifier: Apache-2.0
import json
import os
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Optional

from overrides import override  # type: ignore[ancereportUnknownVariableType]

from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.exporter.base_json_exporter import BaseJsonExporter
from nv_one_logger.exporter.exporter import ExportError

_logger = get_logger(__name__)


class FileExporter(BaseJsonExporter):
    """Exporter implementation that writes spans and events to a file."""

    def __init__(self, file_path: Path):
        """Initialize the FileExporter with a file path.

        Args:
            file_path: The path to the file to write the spans and events to.
        """
        super().__init__()
        self._filepath: Path = file_path
        self._file: Optional[TextIOWrapper] = None

    @override
    def initialize(self) -> None:
        """Initialize the FileExporter.

        Extends the base class implementation for the FileExporter.
        """
        super().initialize()
        self._file = self._open_file()
        self._last_start_span = None

    @override
    def close(self) -> None:
        """Close the FileExporter.

        Extends the base class implementation for the FileExporter.
        """
        if not self._file:
            return
        super().close()
        # This should be done AFTER super.close() because super.close() may need to write the last record.
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

    def _open_file(self) -> TextIOWrapper:
        dir = os.path.dirname(self._filepath)
        if dir:
            os.makedirs(dir, exist_ok=True)
        return open(self._filepath, "w")

    @override
    def _write(self, json_dict: Dict[str, Any]) -> None:
        """Write the given JSON-compatible dictionary to the file."""
        if not self._file:
            raise ExportError("File exporter not initialized or already closed!")

        json_str = json.dumps(json_dict, separators=(",", ":"))
        with self._lock:
            try:
                self._file.write(json_str + "\n")
                self._file.flush()
            except Exception as e:
                _logger.error(f"Error writing record to file: {e}")
