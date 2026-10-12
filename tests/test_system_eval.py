"""Tests for the routing decision (who answers one question)."""

from prism.data import Question
from prism.system_eval import curve_points, route_one


def test_sure_model_answers():
    assert route_one(0.50, gate=0.13, question_has_negation=False, negation_rule_on=True) == "model"


def test_unsure_goes_to_llm():
    assert (
        route_one(0.05, gate=0.13, question_has_negation=False, negation_rule_on=True)
        == "llm (unsure)"
    )


def test_negation_goes_to_llm_first():
    # even a very sure model is overruled by the negation rule
    assert (
        route_one(0.90, gate=0.13, question_has_negation=True, negation_rule_on=True)
        == "llm (negation rule)"
    )


def test_negation_ignored_if_rule_off():
    assert route_one(0.90, gate=0.13, question_has_negation=True, negation_rule_on=False) == "model"


def q(id, text, *labels):
    return Question(id=id, text=text, labels=frozenset(labels), question_type="", part="real-world")


QUESTIONS = [
    q("a", "is the bug fixed?", "jira"),
    q("b", "not the email, what did people say?", "slack"),
]
MODEL_ANSWERS = {"a": frozenset({"jira"}), "b": frozenset({"gmail"})}
CONFIDENCES = {"a": 0.6, "b": 0.7}
LLM_ANSWERS = {"a": frozenset({"jira"}), "b": frozenset({"slack"})}
LLM_COSTS = {"a": 0.002, "b": 0.002}


def test_curve_gate_zero_model_answers_all_but_negation():
    point = curve_points(
        QUESTIONS, MODEL_ANSWERS, CONFIDENCES, LLM_ANSWERS, LLM_COSTS, [0.0], True
    )[0]
    assert point["model_share"] == 0.5  # "b" has a negation word → the LLM, even at gate 0
    assert point["all_exact"] == 1.0  # model right on "a", LLM right on "b"


def test_curve_very_high_gate_llm_answers_everything():
    point = curve_points(
        QUESTIONS, MODEL_ANSWERS, CONFIDENCES, LLM_ANSWERS, LLM_COSTS, [0.99], True
    )[0]
    assert point["model_share"] == 0.0
    assert point["cost_per_million"] == 2000.0  # $0.002 per question × 1,000,000
