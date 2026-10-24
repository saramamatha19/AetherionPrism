"""Tests for the API, with FastAPI's test client (no server, no real LLM, no real log)."""

import pytest
from fastapi.testclient import TestClient
from fakes import FakeClient, fake_reply
from test_router import FakeLog

from prism.api import create_app, load_api_settings
from prism.config import load_settings
from prism.intakeStore import IntakeStore
from prism.registry import read_champion
from prism.router import Router


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    def make_router():
        settings = load_settings().model_copy(update={"mode": "model_only"})
        fake_llm = FakeClient(reply=fake_reply('{"sources": ["slack"]}'))
        return Router(settings=settings, llm_client=fake_llm, log=FakeLog())

    def make_store():
        return IntakeStore(tmp_path_factory.mktemp("intake") / "prism.db")  # never the real file

    with TestClient(
        create_app(make_router, make_store)
    ) as test_client:  # "with" runs the start-up (lifespan)
        yield test_client


def test_health(client):
    reply = client.get("/health").json()
    assert reply["status"] == "ok"
    assert reply["model"] == read_champion()["version"]


def test_route_returns_the_record(client):
    reply = client.post("/route", json={"question": "look at my email"})
    assert reply.status_code == 200
    record = reply.json()
    assert record["answer"] == ["gmail"]
    assert record["decided_by"] == "model" and record["reason"] == "model only"


def test_wrong_mode_is_refused(client):
    reply = client.post("/route", json={"question": "is the bug fixed?", "mode": "gatd"})
    assert reply.status_code == 422  # refused by the RouteRequest shape


def test_empty_question_is_refused(client):
    reply = client.post("/route", json={"question": "   "})
    assert reply.status_code == 400


def test_prism_mode_replaces_the_yaml_mode(monkeypatch):
    monkeypatch.setenv("PRISM_MODE", "model_only")  # what Docker sets
    assert load_api_settings().mode == "model_only"


def test_no_prism_mode_keeps_the_yaml_mode(monkeypatch):
    monkeypatch.delenv("PRISM_MODE", raising=False)
    assert load_api_settings().mode == load_settings().mode


def test_wrong_prism_mode_is_refused(monkeypatch):
    monkeypatch.setenv("PRISM_MODE", "modelonly")
    with pytest.raises(ValueError, match="PRISM_MODE"):
        load_api_settings()
