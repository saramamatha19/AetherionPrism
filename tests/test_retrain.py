"""Tests for collecting reviewed shadow questions into a training batch (throwaway folders)."""

import csv
import json

import pytest

from prism.data import load_questions
from prism.retrain import collect


def make_run(run_dir, decisions, reviewed):
    """A fake shadow run: decisions_*.jsonl + review.csv."""
    run_dir.mkdir()
    with open(run_dir / "decisions_2026-11-01.jsonl", "w") as file:
        for d in decisions:
            line = {"mode": "shadow", "would_escalate": False}
            line.update(d)
            file.write(json.dumps(line) + "\n")
    with open(run_dir / "review.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "question", "model_answer", "llm_answer", "correct_answer"])
        for question, model, llm, correct in reviewed:
            writer.writerow(["disagreement", question, model, llm, correct])


def batch_rows(summary):
    with open(summary["file"], newline="") as file:
        return list(csv.DictReader(file))


def test_collect_uses_review_and_agreements(tmp_path):
    decisions = [
        {
            "question": "where is the zebra onboarding doc?",
            "model_answer": ["kb"],
            "llm_answer": ["kb"],
        },
        {"question": "zebra outage chat?", "model_answer": ["jira"], "llm_answer": ["slack"]},
        {"question": "zebra invoice thread?", "model_answer": ["kb"], "llm_answer": ["gmail"]},
    ]
    reviewed = [
        ("zebra outage chat?", "jira", "slack", "slack")
    ]  # the 3rd disagreement isn't reviewed
    make_run(tmp_path / "run", decisions, reviewed)
    summary = collect(tmp_path / "run", "Sara", new_labels_dir=tmp_path / "new_labels")
    assert summary["from_agreement"] == 1 and summary["from_review"] == 1
    assert summary["skipped_unreviewed"] == 1
    labels = {}
    for row in batch_rows(summary):
        labels[row["question"]] = row["labels"]
    assert labels == {"where is the zebra onboarding doc?": "kb", "zebra outage chat?": "slack"}


def test_collect_refuses_eval_questions(tmp_path):
    eval_question = load_questions("data/eval.csv")[0].text
    make_run(
        tmp_path / "run", [{"question": eval_question, "model_answer": [], "llm_answer": []}], []
    )
    with pytest.raises(ValueError, match="EVAL questions"):
        collect(tmp_path / "run", "Sara", new_labels_dir=tmp_path / "new_labels")


def test_collect_skips_duplicates(tmp_path):
    train_question = load_questions("data/train.csv")[0].text
    decisions = [
        {"question": train_question.upper(), "model_answer": ["jira"], "llm_answer": ["jira"]}
    ]
    make_run(tmp_path / "run", decisions, [])
    summary = collect(tmp_path / "run", "Sara", new_labels_dir=tmp_path / "new_labels")
    assert summary["skipped_duplicate"] == 1 and summary["kept"] == 0


def test_train_stops_without_enough_new_rows(tmp_path):
    from prism.config import load_settings
    from prism.retrain import train_challenger

    with pytest.raises(SystemExit, match="not enough new data"):
        train_challenger(load_settings(), new_labels_dir=tmp_path)  # no batches at all
