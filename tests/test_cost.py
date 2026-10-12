"""Tests for LLM cost, worked out by hand."""

import pytest

from prism.config import load_settings
from prism.cost import llm_call_cost

SETTINGS = load_settings()  # gpt-4o: $2.50 input, $1.25 cached input, $10.00 output, per million


def test_cost_by_hand():
    # 318 × 2.50 / 1,000,000 + 7 × 10.00 / 1,000,000 = 0.000795 + 0.000070
    assert llm_call_cost(318, 0, 7, SETTINGS) == pytest.approx(0.000865)


def test_cached_tokens_are_cheaper():
    all_normal = llm_call_cost(1000, 0, 0, SETTINGS)
    all_cached = llm_call_cost(1000, 1000, 0, SETTINGS)
    assert all_cached == pytest.approx(all_normal / 2)


def test_model_without_prices_is_an_error():
    no_price = SETTINGS.model_copy(update={"llm_model": "some-unknown-model"})
    with pytest.raises(ValueError, match="No prices"):
        llm_call_cost(100, 0, 5, no_price)
