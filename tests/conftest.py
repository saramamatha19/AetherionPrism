"""Shared test fixtures: pytest gives these to every test file automatically (no import needed).

app_client  the API with the real model (model_only, no LLM), loaded ONCE for all tests
client      the same API, with a fresh empty intake store (a temporary file) for each test
"""

import pytest
from fastapi.testclient import TestClient
from test_router import FakeLog

from prism.api import create_app
from prism.config import load_settings
from prism.intakeStore import IntakeStore
from prism.router import Router


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    def make_router():
        settings = load_settings().model_copy(update={"mode": "model_only"})
        return Router(settings=settings, log=FakeLog())

    def make_store():
        return IntakeStore(tmp_path_factory.mktemp("intake") / "prism.db")

    with TestClient(create_app(make_router, make_store)) as test_client:
        yield test_client


@pytest.fixture
def client(app_client, tmp_path):
    app_client.app.state.store = IntakeStore(tmp_path / "prism.db")  # a fresh, empty store
    return app_client
