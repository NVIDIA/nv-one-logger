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
        logger_config: LoggerConfig = OneLoggerProvider.instance().config.logger_config
        formatter = logging.Formatter(logger_config.log_format)
        fh_info = logging.FileHandler(logger_config.log_file_path_for_info)
        fh_info.setLevel(logging.INFO)
        fh_info.setFormatter(formatter)
        logger.addHandler(fh_info)

        fh_err = logging.FileHandler(logger_config.log_file_path_for_err)
        fh_err.setLevel(logging.ERROR)
        fh_err.setFormatter(formatter)
        logger.addHandler(fh_err)

        logger.setLevel(logger_config.log_level)

    return logger
