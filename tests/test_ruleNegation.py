"""Tests for the negation rule."""

from prism.ruleNegation import has_negation


def test_negation_words_are_found():
    assert has_negation("not the email, what did people say in slack")
    assert has_negation("anything except jira")
    assert has_negation("Don't use the FAQ")  # capital letters are fine


def test_similar_words_are_not_negation():
    assert not has_negation("what's in my notion notes")  # "notion" / "notes" are not "not"
    assert not has_negation("is the bug fixed?")
