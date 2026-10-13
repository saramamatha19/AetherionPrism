"""Tests for the shadow report, on a tiny made-up log worked out by hand."""

import csv

import pytest

from prism.config import load_settings
from prism.shadowReport import go_live_check, read_review_file, shadow_report


def d(question, model, llm, escalate):
    return {
        "question": question,
        "model_answer": model,
        "llm_answer": llm,
        "would_escalate": escalate,
    }


DECISIONS = [
    d("q1", ["jira"], ["jira"], False),  # confident, agree
    d("q2", ["jira"], ["jira", "slack"], False),  # confident, disagree: model misses slack
    d("q3", ["kb"], ["web"], True),  # NOT confident → doesn't count for go-live
    d("q1", ["jira"], ["jira"], False),  # a repeat of q1, confident, agree
]


def test_shadow_report_by_hand():
    report = shadow_report(DECISIONS, spot_check_rate=0.0, seed=42)
    assert report["agreement_all"] == 0.5  # q1, q1 agree; q2, q3 don't → 2/4
    assert report["confident"] == 3  # q1, q2, q1
    assert report["agreement_confident"] == 2 / 3
    assert report["model_misses"]["slack"] == 1
    assert len(report["to_review"]) == 1  # only q2 (the confident disagreement)
    assert report["repeat_share"] == 0.25  # 1 repeat out of 4


def test_spot_check_samples_agreements():
    report = shadow_report(DECISIONS, spot_check_rate=1.0, seed=42)
    kinds = []
    for item in report["to_review"]:
        kinds.append(item["kind"])
    assert kinds.count("spot-check") == 2  # both confident agreements are checked


def test_go_live_by_hand():
    # 10 confident questions: 8 agreements, 2 disagreements (+ the reviewed rows below)
    report = {"questions": 600, "confident": 10, "confident_agreements": 8}
    reviewed = [
        {
            "kind": "spot-check",
            "model_answer": "jira",
            "llm_answer": "jira",
            "correct_answer": "jira",
        },
        {"kind": "spot-check", "model_answer": "kb", "llm_answer": "kb", "correct_answer": "kb"},
        {
            "kind": "disagreement",
            "model_answer": "slack",
            "llm_answer": "web",
            "correct_answer": "slack",
        },
        {
            "kind": "disagreement",
            "model_answer": "kb",
            "llm_answer": "web",
            "correct_answer": "none",
        },
    ]
    result = go_live_check(report, reviewed, load_settings())
    # agreements 100% right (2/2) → 8 right; model right on 1 disagreement → 9/10 = 90%
    assert result["model_accuracy"] == 0.9
    assert result["llm_accuracy"] == 0.8  # 8 + 0 → 80%
    assert result["verdict"] == "GO"  # 90% ≥ 85%, ≥ LLM, 600 ≥ 500


def test_unreviewed_row_is_an_error(tmp_path):
    path = tmp_path / "review.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "question", "model_answer", "llm_answer", "correct_answer"])
        writer.writerow(["disagreement", "is the bug fixed?", "jira", "slack", ""])
    with pytest.raises(ValueError, match="Not reviewed yet"):
        read_review_file(path)
