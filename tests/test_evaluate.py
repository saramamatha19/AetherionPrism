"""Tests for evaluate(): a 2-question exam worked out by hand."""

from prism.data import Question
from prism.evaluate import evaluate

QUESTIONS = [
    Question(
        id="a",
        text="is the bug fixed?",
        labels=frozenset({"jira"}),
        question_type="Implied source",
        part="real-world",
    ),
    Question(
        id="b",
        text="thanks!",
        labels=frozenset(),
        question_type="Small talk",
        part="real-world",
    ),
]


def always_jira(text):
    return frozenset({"jira"})


def test_evaluate_marks_and_groups():
    result = evaluate(always_jira, QUESTIONS)
    assert result["all"]["exact_match"] == 0.5  # right on a, wrong on b
    assert result["all"]["found_all_needed"] == 1.0  # nothing needed in b → nothing missed
    assert result["by_type"]["Small talk"]["exact_match"] == 0.0
    assert result["by_type"]["Implied source"]["exact_match"] == 1.0
