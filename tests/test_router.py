"""Tests for the live router, with a FAKE LLM (free; never a real API call).

The router loads model B once for all tests (it takes a while), so these tests need
models/tfidf_ftv2_bge_logreg_v2 and finetuned/bge_ft_v2 on this laptop.
"""

import openai
import pytest
from fakes import REQUEST, FakeClient, fake_reply

from prism.config import load_settings
from prism.router import Router

SURE_QUESTION = "look at my email"  # model B: gmail, very sure
NEGATION_QUESTION = "not the email, what did people say in slack about the outage?"


class FakeLog:
    """Collects records instead of writing files."""

    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class CountingClient(FakeClient):
    """A fake LLM that always answers {slack} and counts its calls."""

    def __init__(self):
        super().__init__(reply=fake_reply('{"sources": ["slack"]}'))
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return super().create(**kwargs)


@pytest.fixture(scope="module")
def router():
    """One router for all tests (loading the model once); each test sets the mode it needs."""
    settings = load_settings().model_copy(update={"mode": "gated", "audit_rate": 0.0})
    return Router(settings=settings, llm_client=CountingClient(), log=FakeLog())


def use(router, mode, audit_rate=0.0, client=None):
    """Switch the shared router to a mode, a fresh fake LLM and an empty fake log."""
    router.mode = mode
    router.settings = router.settings.model_copy(update={"audit_rate": audit_rate})
    router.llm_client = client or CountingClient()
    router.log = FakeLog()
    return router


def test_warm_up_is_not_logged():
    log = FakeLog()
    settings = load_settings().model_copy(update={"mode": "model_only"})
    Router(settings=settings, log=log)  # creating it runs the warm-up
    assert log.records == []


def test_model_only_never_calls_the_llm(router):
    use(router, "model_only")
    record = router.route(NEGATION_QUESTION)
    assert router.llm_client.calls == 0
    assert record.decided_by == "model" and record.reason == "model only"


def test_shadow_always_asks_llm_and_keeps_the_model_answer(router):
    use(router, "shadow")
    record = router.route(SURE_QUESTION)
    assert router.llm_client.calls == 1
    assert record.answer == {"slack"}  # the user gets the LLM's answer
    assert record.model_answer == {"gmail"}  # the model's answer is kept
    assert record.would_escalate is False  # gated mode would have let the model answer


def test_gated_sure_question_model_answers(router):
    use(router, "gated")
    record = router.route(SURE_QUESTION)
    assert router.llm_client.calls == 0
    assert record.reason == "model sure" and record.answer == {"gmail"}


def test_gated_negation_goes_to_llm(router):
    use(router, "gated")
    record = router.route(NEGATION_QUESTION)
    assert record.reason == "negation rule" and record.decided_by == "llm"


def test_llm_failure_uses_the_backup_plan(router):
    failing = FakeClient(error=openai.APITimeoutError(request=REQUEST))
    use(router, "shadow", client=failing)
    record = router.route(SURE_QUESTION)
    assert record.reason == "llm failed → backup"
    assert record.llm_error == "timeout"
    assert record.answer == {"gmail"}  # the model's answer as the backup


def test_audit_keeps_the_model_answer(router):
    use(router, "gated", audit_rate=1.0)  # audit every "model sure" question
    record = router.route(SURE_QUESTION)
    assert record.reason == "audit"
    assert record.decided_by == "model" and record.answer == {"gmail"}  # the user gets the model's
    assert record.llm_answer == {"slack"}  # the LLM's answer is only recorded


def test_every_decision_is_logged(router):
    use(router, "gated")
    router.route(SURE_QUESTION)
    router.route(NEGATION_QUESTION)
    assert len(router.log.records) == 2
