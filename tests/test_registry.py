"""Tests for saving and loading models (in a throwaway folder, never the real models/)."""

from prism.config import load_settings
from prism.data import Question
from prism.logreg import scores, train
from prism.registry import load_model, save_model
from prism.tfidf import build_tfidf


def q(text, *labels):
    return Question(id=text, text=text, labels=frozenset(labels), question_type="", part="")


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


def test_save_then_load_gives_same_scores(tmp_path):
    model = train(TINY, features=build_tfidf())
    thresholds = {"jira": 0.3}
    version = save_model(
        model, thresholds, "test", load_settings(), FAKE_RESULT, models_dir=tmp_path
    )
    loaded_model, loaded_thresholds = load_model(version, models_dir=tmp_path)
    assert scores(loaded_model, "is the bug fixed") == scores(model, "is the bug fixed")
    assert loaded_thresholds == thresholds


def test_versions_go_up_and_never_overwrite(tmp_path):
    model = train(TINY, features=build_tfidf())
    first = save_model(model, {}, "test", load_settings(), FAKE_RESULT, models_dir=tmp_path)
    second = save_model(model, {}, "test", load_settings(), FAKE_RESULT, models_dir=tmp_path)
    assert first == "test_v1"
    assert second == "test_v2"
