"""Tests for confidence, using the examples worked out by hand."""

import pytest

from prism.confidence import confidence, how_sure


def test_how_sure_one_judge():
    assert how_sure(0.00, 0.25) == 1.0  # certain "no"
    assert how_sure(0.20, 0.25) == pytest.approx(0.20)  # a little "no"
    assert how_sure(0.25, 0.25) == 0.0  # coin flip
    assert how_sure(0.40, 0.25) == pytest.approx(0.20)  # a little "yes"
    assert how_sure(1.00, 0.25) == 1.0  # certain "yes"


def test_confidence_is_the_least_sure_judge():
    scores = {"jira": 0.59, "web": 0.22, "slack": 0.05}
    thresholds = {"jira": 0.30, "web": 0.25, "slack": 0.25}
    sure, weakest = confidence(scores, thresholds)
    assert weakest == "web"
    assert sure == pytest.approx(0.12)  # (0.25 − 0.22) ÷ 0.25
