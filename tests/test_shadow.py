"""Tests for the replay LLM used by the shadow rehearsal (reads the real eval recording)."""

import json

import pytest

from prism.config import load_settings
from prism.data import load_questions
from prism.llm_answer_eval import answers_file, read_answers
from prism.shadow import ReplayLLMClient

SETTINGS = load_settings()
QUESTIONS = load_questions("data/eval.csv")[:3]


def ask(client, text):
    return client.create(
        model="x", messages=[{"role": "system", "content": "…"}, {"role": "user", "content": text}]
    )


def test_replay_client_returns_the_recorded_answer():
    client = ReplayLLMClient(QUESTIONS, SETTINGS)
    recorded = read_answers(answers_file(SETTINGS, "eval"))[QUESTIONS[0].id]
    reply = ask(client, QUESTIONS[0].text)
    assert json.loads(reply.choices[0].message.content)["sources"] == recorded["llm_sources"]
    assert reply.usage.prompt_tokens == recorded["prompt_tokens"]  # real tokens → real cost


def test_unrecorded_question_is_an_error():
    client = ReplayLLMClient(QUESTIONS, SETTINGS)
    with pytest.raises(KeyError):
        ask(client, "a question that was never recorded")
