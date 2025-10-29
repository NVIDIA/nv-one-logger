# SPDX-License-Identifier: Apache-2.0
import logging

from nv_one_logger.api.config import LoggerConfig
from nv_one_logger.api.one_logger_provider import OneLoggerProvider


def get_logger(name: str) -> logging.Logger:
    """Initialize a Python logger based on the user configuration.

    Args:
        name: Name of the logger

    Returns:
        logging.Logger: Configured Python logger instance
    """
    logger = logging.getLogger(name)

    if OneLoggerProvider.instance().one_logger_ready:
        # Check if handlers are already added to avoid recreating them
        if not logger.handlers:
            try:
                logger_config: LoggerConfig = OneLoggerProvider.instance().config.logger_config

                # Always set log level (even when disabled for rank, to suppress internal logs)
                logger.setLevel(logger_config.log_level)

                # Only set up file handlers if OneLogger is enabled for current rank
                if OneLoggerProvider.instance().one_logger_enabled:
                    formatter = logging.Formatter(logger_config.log_format)

                    fh_info = logging.FileHandler(logger_config.log_file_path_for_info)
                    fh_info.setLevel(logging.INFO)
                    fh_info.setFormatter(formatter)
                    logger.addHandler(fh_info)

                    fh_err = logging.FileHandler(logger_config.log_file_path_for_err)
                    fh_err.setLevel(logging.ERROR)
                    fh_err.setFormatter(formatter)
                    logger.addHandler(fh_err)
            except OSError:
                # File logging failed due to I/O issues. Add NullHandler to respect user's intent
                # (they configured file-only logging, so we don't want to fall back to stderr).
                logger.addHandler(logging.NullHandler())
            except Exception:
                # Any other error (config access, formatter creation, etc.) - let logger propagate normally
                # This allows internal OneLogger warnings to be logged via default Python logging
                pass

    return logger
