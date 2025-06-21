# SPDX-License-Identifier: Apache-2.0
import json
import logging
import re
from typing import Any, Dict, List

from overrides import override  # type: ignore[ancereportUnknownVariableType]

from nv_one_logger.core.internal.logging import get_logger
from nv_one_logger.exporter.base_json_exporter import BaseJsonExporter
from nv_one_logger.exporter.exporter import ExportError

_logger = get_logger(__name__)


class LoggerExporter(BaseJsonExporter):
    """Exporter implementation that writes spans and using a python logger.

    Events are formatted as key=value pairs with hierarchical keys separated by dots.
    """

    def __init__(self, logger: logging.Logger):
        super().__init__()
        self._logger = logger

    @override
    def initialize(self) -> None:
        """Initialize the LoggerExporter.

        Extends the base class implementation for the LoggerExporter.
        See the docstrings of the Exporter interface for more details.
        """
        super().initialize()

    def _format_log(self, json_dict: Dict[str, Any], prefix: str = "") -> str:
        """
        Format the dictionary into a log string. The output string will be in the format of "[key1=value1 | key2=value2 | ...]".

        Nested dictionaries are formatted recursively, and will be in the format of "key1.key2=value where key1 is the top
        level key and key2 is the nested key in the nested dictionary".

        The prefix argument indicates if the dictionary is a top-level dictionary or a nested dictionary.
        Top level dictionaries are enclosed in square brackets, and have some additional properties like the process id and
        the record count added to the dictionary.

        Args:
            event_json (dict): The dictionary to format.
            prefix (str): The prefix indicates if the dictionary is a top-level dictionary or a nested dictionary.
                          This function is in fact called recursively to format nested dictionaries, and for nested dictionaries
                          the prefix is the key of the nested dictionary in the parent dictionary.

        Limitations:
            - dictionary keys cannot contain dots.
        Returns:
            str: The formatted log string.
        """
        items: List[str] = []
        for k, v in sorted(json_dict.items(), reverse=True):
            if "." in k:
                raise ExportError(f"keys cannot contain dots: {k}")

            if isinstance(v, dict):
                child_prefix = f"{prefix}.{k}" if prefix else k
                formatted_dict = self._format_log(json_dict=v, prefix=child_prefix)
                if formatted_dict:
                    items.append(formatted_dict)
            else:
                # We are not producing json but this ensures the strings are encoded with double quotes.
                encoded_value = json.dumps(v)
                if prefix:
                    items.append(f"{prefix}.{k}={encoded_value}")
                else:
                    items.append(f"{k}={encoded_value}")
        if prefix:
            return " | ".join(items) if items else ""
        else:
            return "[" + " | ".join(items) + "]" if items else ""

    @staticmethod
    def deserialize_log(text: str) -> Dict[str, Any]:
        """
        Unformat a log string back into a dictionary by reversing the format_log function.

        Refer to that functionfor details on the format of the log string.
        """
        # First extract all the characters that are in the top level square brackets.
        pattern = r"\[(.*)\]"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            raise ExportError(f"Log string must be enclosed in square brackets: {text}")

        # Then split the text by the pipe character with a space on each side of the pipe.
        items = matches[-1].strip().rstrip("|").split(" | ")  # FIXME - exclude text inside double quotes

        ret: Dict[str, Any] = {}
        for item in items:
            k, v = item.split("=")  # FIXME - exclude text inside double quotes
            d = ret
            while "." in k:
                k1, k2 = k.split(".", 1)
                if k1 not in d:
                    d[k1] = {}
                k = k2
                d = d[k1]
            try:
                d[k] = json.loads(v)
            except Exception as e:
                _logger.error(f"Failed to deserialize log string: {e}")
                d[k] = v
        return ret

    @override
    def _write(self, json_dict: Dict[str, Any]) -> None:
        """Write the given JSON-compatible dictionary to the logger."""
        json_str = self._format_log(json_dict)
        self._logger.info(json_str)
