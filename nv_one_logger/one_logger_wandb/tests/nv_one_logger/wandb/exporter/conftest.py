# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(autouse=True)
def mock_get_current_time_msec(monkeypatch):
    """Mock get_current_time_msec to return a fixed value for all tests."""
    from nv_one_logger.wandb.exporter import wandb_exporter

    monkeypatch.setattr(wandb_exporter, "get_current_time_msec", lambda: 1000)
    yield
