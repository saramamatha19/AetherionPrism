"""Tests for the marking rules, using a mini exam worked out by hand."""

import pytest

from prism.metrics import exact_match, found_all_needed, per_source_scores

# The mini exam: (answer key, student said)
TRUTHS = [
    frozenset({"jira"}),  # Q1
    frozenset({"jira", "confluence"}),  # Q2
    frozenset({"slack"}),  # Q3
    frozenset(),  # Q4: no search needed
]
PREDICTIONS = [
    frozenset({"jira"}),  # Q1: right
    frozenset({"jira"}),  # Q2: missed confluence
    frozenset({"slack", "gmail"}),  # Q3: extra gmail
    frozenset({"slack"}),  # Q4: searched when not needed
]


# 1. Exact match: only Q1 is fully right → 1 out of 4
def test_exact_match():
    results = [exact_match(t, p) for t, p in zip(TRUTHS, PREDICTIONS)]
    assert results == [True, False, False, False]


# 2. All needed found: only Q2 missed something → 3 out of 4
def test_found_all_needed():
    results = [found_all_needed(t, p) for t, p in zip(TRUTHS, PREDICTIONS)]
    assert results == [True, False, True, True]


# 3. Order never matters
def test_order_does_not_matter():
    assert exact_match(frozenset({"jira", "slack"}), frozenset({"slack", "jira"}))


# 4. Per-source marks match what we worked out by hand
def test_per_source_scores():
    scores = per_source_scores(TRUTHS, PREDICTIONS)
    assert scores["jira"]["recall"] == 1.0  # needed 2×, picked 2×
    assert scores["jira"]["precision"] == 1.0  # picked 2×, right 2×
    assert scores["slack"]["precision"] == 0.5  # picked 2×, right 1×
    assert scores["slack"]["f1"] == pytest.approx(0.667, abs=0.001)
    assert scores["confluence"]["recall"] == 0.0  # needed 1×, never picked
