# SPDX-License-Identifier: Apache-2.0
"""This module provides functions to get the version number of the package."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import toml

from nv_one_logger.core.internal.logging import get_logger

_logger = get_logger(__name__)


def find_pyproject(start_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the pyproject.toml file by traversing up the directory tree."""
    if start_dir is None:
        start_dir = Path(__file__).resolve().parent

    current_dir = Path(start_dir)
    while current_dir != current_dir.parent:
        pyproject_path = current_dir / "pyproject.toml"
        if pyproject_path.is_file():
            return pyproject_path
        current_dir = current_dir.parent

    return None


def get_version_from_pyproject() -> Optional[str]:
    """Get the version number from pyproject.toml file."""
    pyproject_path = find_pyproject()
    if pyproject_path is None:
        print("Error: pyproject.toml file not found.")
        return None

    try:
        with open(pyproject_path, "r") as f:
            data = toml.load(f)

        version = data.get("project", {}).get("version", None)
        if version:
            return str(version)

        version = data.get("tool", {}).get("poetry", {}).get("version", None)
        return str(version) if version else None

    except Exception as e:
        _logger.error(f"Error reading version from pyproject.toml: {str(e)}")
        return None


def get_version(pkg_name: str) -> str:
    """Get the version number from pyproject.toml file.

    NOTE: importlib.metadata only supports get version from
          installed package. so we just read the version number from toml file.
          see: https://github.com/python-poetry/poetry/issues/273#issuecomment-726059226

    :return: version number. {major}.{minor}.{patch}
    :rtype: str
    """
    # Try to get version from importlib.metadata
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        pass

    # Try to get version from toml file
    ver = get_version_from_pyproject()
    if ver is None:
        _logger.error(f"Version not found for package {pkg_name}")
        return "unknown"
    return ver
