"""Tests for the LLM failure report, worked out by hand."""

from prism.data import Question
from prism.llm_failures import failure_report


def q(id, *labels):
    return Question(id=id, text=id, labels=frozenset(labels), question_type="", part="")


QUESTIONS = [q("a", "kb"), q("b", "kb"), q("c", "jira")]
LLM_ANSWERS = {
    "a": frozenset({"kb", "confluence"}),  # added confluence
    "b": frozenset({"kb", "confluence"}),  # added confluence again
    "c": frozenset({"jira"}),  # right
}


def test_failure_report_by_hand():
    report = failure_report(QUESTIONS, LLM_ANSWERS)
    assert report["added_wrongly"]["confluence"] == 2
    assert report["missed"]["kb"] == 0
    assert report["confusions"][(("kb",), ("confluence", "kb"))] == 2
    assert len(report["failed_examples"]) == 2  # "c" was right, so not listed
