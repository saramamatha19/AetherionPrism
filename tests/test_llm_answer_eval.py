"""Tests for recording and replaying LLM answers (fake LLM, throwaway folder: free, no real files)."""

import openai
import pytest
from fakes import REQUEST, FakeClient, fake_reply

from prism.config import load_settings
from prism.data import Question
from prism.llm_answer_eval import load_llm_answers, prompt_fingerprint, record_llm_answers

SETTINGS = load_settings()


def q(id, text):
    return Question(id=id, text=text, labels=frozenset(), question_type="", part="")


QUESTIONS = [q("a", "is the bug fixed?"), q("b", "what did people say in chat?")]


class CountingClient(FakeClient):
    """A fake LLM that counts how many times it was asked."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return super().create(**kwargs)


def test_resume_does_not_ask_again(tmp_path):
    client = CountingClient(reply=fake_reply('{"sources": ["jira"]}'))
    record_llm_answers(QUESTIONS, SETTINGS, client=client, recordings_dir=tmp_path)
    record_llm_answers(QUESTIONS, SETTINGS, client=client, recordings_dir=tmp_path)
    assert client.calls == 2  # 2 questions asked once each; the 2nd run asked nothing


def test_failed_questions_are_retried(tmp_path):
    failing = FakeClient(error=openai.APITimeoutError(request=REQUEST))
    record_llm_answers(QUESTIONS, SETTINGS, client=failing, recordings_dir=tmp_path)
    working = CountingClient(reply=fake_reply('{"sources": ["slack"]}'))
    record_llm_answers(QUESTIONS, SETTINGS, client=working, recordings_dir=tmp_path)
    assert working.calls == 2  # both failed before → both retried
    answers = load_llm_answers(QUESTIONS, SETTINGS, recordings_dir=tmp_path)
    assert answers["a"] == {"slack"}  # the retry replaced the failure


def test_missing_answer_is_an_error(tmp_path):
    client = FakeClient(reply=fake_reply('{"sources": ["jira"]}'))
    record_llm_answers(QUESTIONS[:1], SETTINGS, client=client, recordings_dir=tmp_path)
    with pytest.raises(ValueError, match="no LLM answer"):
        load_llm_answers(QUESTIONS, SETTINGS, recordings_dir=tmp_path)  # "b" was never asked


def test_prompt_change_means_new_file():
    assert prompt_fingerprint("prompt one") != prompt_fingerprint("prompt two")
