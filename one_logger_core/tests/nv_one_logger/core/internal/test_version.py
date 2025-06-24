from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

from nv_one_logger.core.internal.version import get_version


@patch("nv_one_logger.core.internal.version.version")
def test_get_version_from_installed_package(mock_version: MagicMock) -> None:
    """Test that the version is read from importlib.metadata if the package is installed."""
    mock_version.return_value = "1.2.3"

    version = get_version("nv-one-logger-core")
    assert version == "1.2.3"
    mock_version.assert_called_once_with("nv-one-logger-core")


@patch("nv_one_logger.core.internal.version.version")
def test_get_version_from_toml(mock_version: MagicMock) -> None:
    """Test that the version is read from pyproject.toml if the package is not installed."""
    mock_version.side_effect = PackageNotFoundError
    version = get_version("nv-one-logger-core")
    assert version == "2.0.0"
    mock_version.assert_called_once_with("nv-one-logger-core")
