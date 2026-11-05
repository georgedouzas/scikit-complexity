"""Test the `_heuristics` module of `skcomplex.systems.economics.common.forecasting`."""

from skcomplex.systems.economics.common.forecasting import heuristics


def test_heuristics_follow_their_formulas():
    """Forecasting heuristics follow their formulas."""
    assert heuristics.adaptive([2.0], 1.0, weight=0.5) == 1.5
    assert heuristics.trend([1.0, 2.0], strength=1.0) == 3.0
    assert heuristics.anchoring([1.0, 3.0]) == 4.5
    assert heuristics.fundamental([3.0], fundamental=1.0, reversion=0.5) == 2.0
