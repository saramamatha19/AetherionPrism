"""Tests for the comparison helpers."""

from prism.compare import percentile


def test_percentile():
    values = []
    for i in range(1, 101):  # 1, 2, ... 100
        values.append(i)
    assert percentile(values, 50) == 51
    assert percentile(values, 95) == 96
