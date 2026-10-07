"""Tests for the keyword-rules baseline."""

from prism.baselines import keyword_rules


def test_finds_clue_words():
    assert keyword_rules("is the login bug fixed?") == {"jira"}
    assert keyword_rules("did the client reply to the email?") == {"gmail"}


def test_no_clue_means_no_search():
    assert keyword_rules("good morning!") == frozenset()


def test_whole_words_only():
    assert keyword_rules("ask the admin") == frozenset()  # "dm" inside "admin" doesn't count
