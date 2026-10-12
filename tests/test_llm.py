"""Tests for LLM error handling, using a FAKE LLM (free, instant, no real API call)."""

import httpx2
import openai
import pytest
from fakes import REQUEST, FakeClient, fake_reply

from prism.config import load_settings
from prism.data import SOURCES
from prism.llm import ask_llm_safely

SETTINGS = load_settings()


def test_good_reply_is_used():
    client = FakeClient(reply=fake_reply('{"sources": ["slack"]}'))
    result = ask_llm_safely(client, SETTINGS, "what did people say in chat?")
    assert result["sources"] == {"slack"}
    assert result["llm_error"] is None and result["fallback"] is None


def test_timeout_uses_the_model_answer():
    client = FakeClient(error=openai.APITimeoutError(request=REQUEST))
    result = ask_llm_safely(client, SETTINGS, "is the bug fixed?", model_answer={"jira"})
    assert result["sources"] == {"jira"}
    assert result["llm_error"] == "timeout" and result["fallback"] == "model answer"


def test_timeout_without_model_answer_searches_all():
    client = FakeClient(error=openai.APITimeoutError(request=REQUEST))
    result = ask_llm_safely(client, SETTINGS, "is the bug fixed?", model_answer=frozenset())
    assert result["sources"] == frozenset(SOURCES)
    assert result["fallback"] == "all sources"


def test_refusal_is_handled():
    client = FakeClient(reply=fake_reply(None, refusal="I can't help with that."))
    result = ask_llm_safely(client, SETTINGS, "something odd", model_answer={"web"})
    assert result["llm_error"] == "refusal" and result["sources"] == {"web"}


def test_wrong_key_stops_loudly():
    response = httpx2.Response(401, request=REQUEST)
    client = FakeClient(error=openai.AuthenticationError("bad key", response=response, body=None))
    with pytest.raises(openai.AuthenticationError):
        ask_llm_safely(client, SETTINGS, "is the bug fixed?", model_answer={"jira"})
