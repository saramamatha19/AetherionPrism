"""Tests for the decision record: exact JSON for code, readable CSV for people."""

import json

from prism.decisionRecord import CSV_COLUMNS, DecisionRecord


def example_record(**changes) -> DecisionRecord:
    values = {
        "time": "2026-10-13 14:02:11",
        "question": "is the login bug fixed?",
        "answer": frozenset({"jira"}),
        "decided_by": "model",
        "reason": "model sure",
        "model_answer": frozenset({"slack", "jira"}),
        "model_confidence": 0.4123,
        "gate": 0.13,
        "would_escalate": False,
        "llm_answer": None,
        "llm_error": None,
        "cost_spent": 0.0,
        "cost_if_llm": 0.0017,
        "latency_ms": 7.234,
        "mode": "gated",
        "model_version": "tfidf_ftv2_bge_logreg_v2",
        "llm_model": "gpt-4o-2024-11-20",
        "prompt_version": "v4",
        "config_hash": "a91c00",
    }
    values.update(changes)
    return DecisionRecord(**values)


def test_json_keeps_exact_values():
    back = json.loads(example_record().to_json())
    assert back["model_answer"] == ["jira", "slack"]  # a set becomes a sorted list
    assert back["llm_answer"] is None  # None stays null
    assert back["would_escalate"] is False  # True/False stay booleans
    assert back["model_confidence"] == 0.4123  # numbers stay exact


def test_csv_row_is_readable():
    row = dict(zip(CSV_COLUMNS, example_record().to_csv_row()))
    assert row["model_answer"] == "jira|slack"  # joined like our dataset
    assert row["llm_answer"] == ""  # None → empty cell
    assert row["model_confidence"] == "0.41"  # rounded
    assert row["latency_ms"] == "7.2"


def test_csv_and_json_have_the_same_fields():
    record = example_record()
    assert list(json.loads(record.to_json()).keys()) == CSV_COLUMNS
    assert len(record.to_csv_row()) == len(CSV_COLUMNS)
