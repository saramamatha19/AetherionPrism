"""Tests for loading the dataset."""

import pytest

from prism.data import load_questions, parse_labels, split_train


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

# for train -> fit,gate,threshold(4,5)

# 4. Every train question lands in exactly one slice
def test_split_slices_do_not_overlap():
    questions = load_questions("data/train.csv")
    fit, threshold, gate = split_train(questions, 500, 500, seed=42)
    ids = [q.id for q in fit + threshold + gate]
    assert len(ids) == len(set(ids)) == 3219  # every question used exactly once
    assert len(threshold) == 500 and len(gate) == 500


# 5. Same seed → same split (repeatable results)
def test_same_seed_gives_same_split():
    questions = load_questions("data/train.csv")
    first = split_train(questions, 500, 500, seed=42)
    second = split_train(questions, 500, 500, seed=42)
    assert first == second
