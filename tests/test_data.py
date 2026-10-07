"""Tests for loading the dataset."""

import pytest

from prism.data import load_questions, parse_labels


# 1. The frozen eval set is complete (1,200 rows: 500 / 560 / 140)
def test_eval_set_is_complete():
    questions = load_questions("data/eval.csv")
    assert len(questions) == 1200
    assert sum(q.part == "real-world" for q in questions) == 500


# 2. Order doesn't matter, and empty means "no search"
def test_labels_are_sets():
    assert parse_labels("jira|confluence") == parse_labels("confluence|jira")
    assert parse_labels("") == frozenset()


# 3. A misspelled label is an error
def test_unknown_label_is_an_error():
    with pytest.raises(ValueError, match="jria"):
        parse_labels("jria")
