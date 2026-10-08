"""Tests for choosing and applying pass marks."""

from prism.thresholds import apply_thresholds, choose_thresholds


def test_apply_keeps_only_sources_that_pass():
    scores = {"jira": 0.59, "slack": 0.12, "web": 0.22}
    marks = {"jira": 0.30, "slack": 0.35, "web": 0.30}
    assert apply_thresholds(scores, marks) == {"jira"}


def test_choose_finds_the_mark_that_separates_yes_from_no():
    # jira is needed when it scores 0.6 or 0.4, not when it scores 0.2 or 0.1
    all_scores = [{"jira": 0.6}, {"jira": 0.4}, {"jira": 0.2}, {"jira": 0.1}]
    truths = [frozenset({"jira"}), frozenset({"jira"}), frozenset(), frozenset()]
    assert choose_thresholds(all_scores, truths) == {"jira": 0.25}
