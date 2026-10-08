"""Tests for the risk–coverage table, worked out by hand."""

import pytest

from prism.gate import choose_gate, risk_coverage

CONFIDENCES = [0.9, 0.6, 0.3, 0.1]
CORRECT = [True, True, False, False]


def test_risk_coverage_by_hand():
    table = risk_coverage(CONFIDENCES, CORRECT)
    rows = {}
    for row in table:
        rows[row["gate"]] = row

    assert rows[0.0]["kept"] == 4 and rows[0.0]["accuracy"] == 0.5  # keeps all: 2 of 4 right
    assert rows[0.2]["kept"] == 3 and rows[0.2]["accuracy"] == pytest.approx(0.667, abs=0.001)
    assert rows[0.5]["kept"] == 2 and rows[0.5]["accuracy"] == 1.0  # keeps 0.9 and 0.6: both right
    assert table[-1]["gate"] == 0.9  # above 0.9 nothing is kept → table stops


def test_choose_gate_lowest_that_reaches_the_floor():
    table = risk_coverage(CONFIDENCES, CORRECT)
    assert choose_gate(table, 0.60) == 0.11  # keeps 0.9, 0.6, 0.3 → 2 of 3 right = 67%
    assert choose_gate(table, 0.80) == 0.31  # keeps 0.9, 0.6     → 2 of 2 right = 100%


def test_gate_zero_is_refused():
    table = risk_coverage(CONFIDENCES, CORRECT)
    with pytest.raises(ValueError, match="Gate is 0"):
        choose_gate(table, 0.50)  # keeping everything is already 50% → gate 0 → refused


def test_no_gate_good_enough_is_an_error():
    table = risk_coverage([0.9, 0.5], [False, False])  # the model is always wrong
    with pytest.raises(ValueError, match="No gate reaches"):
        choose_gate(table, 0.80)
