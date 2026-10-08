"""Tests for saving and loading models (in a throwaway folder, never the real models/)."""

from prism.config import load_settings
from prism.data import Question
from prism.logreg import scores, train
import joblib
import pytest

from prism.registry import load_model, save_model
from prism.tfidf import build_tfidf


def q(text, *labels):
    return Question(
        id=text, text=text, labels=frozenset(labels), question_type="", part=""
    )


TINY = [
    q("is the bug fixed", "jira"),
    q("open a ticket for the bug", "jira"),
    q("what did people say in chat", "slack"),
    q("check the slack channel", "slack"),
    q("did the client reply to my email", "gmail"),
    q("search my inbox", "gmail"),
]

FAKE_RESULT = {
    "all": {"exact_match": 0.5, "found_all_needed": 0.6},
    "by_part": {"real-world": {"exact_match": 0.7}},
}
FAKE_GATE_RESULT = {"coverage": 0.6, "accuracy": 0.8}


def save(model, thresholds, gate, models_dir):
    """Save with fake results, so each test stays short."""
    return save_model(
        model,
        thresholds,
        gate,
        "test",
        load_settings(),
        FAKE_RESULT,
        FAKE_GATE_RESULT,
        FAKE_GATE_RESULT,
        models_dir=models_dir,
    )


def test_save_then_load_gives_same_scores(tmp_path):
    model = train(TINY, features=build_tfidf())
    thresholds = {"jira": 0.3}
    version = save(model, thresholds, 0.11, tmp_path)
    loaded_model, loaded_thresholds, loaded_gate = load_model(
        version, models_dir=tmp_path
    )
    assert scores(loaded_model, "is the bug fixed") == scores(model, "is the bug fixed")
    assert loaded_thresholds == thresholds
    assert loaded_gate == 0.11


def test_versions_go_up_and_never_overwrite(tmp_path):
    model = train(TINY, features=build_tfidf())
    first = save(model, {}, 0.11, tmp_path)
    second = save(model, {}, 0.11, tmp_path)
    assert first == "test_v1"
    assert second == "test_v2"


def test_model_without_gate_is_refused(tmp_path):
    (tmp_path / "old_v1").mkdir()
    joblib.dump({"model": None, "thresholds": {}}, tmp_path / "old_v1" / "model.joblib")
    with pytest.raises(ValueError, match="has no gate"):
        load_model("old_v1", models_dir=tmp_path)
