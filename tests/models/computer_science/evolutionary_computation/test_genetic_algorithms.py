"""Test the `_genetic_algorithms` module of `skcomplex.models.computer_science.evolutionary_computation`."""

from skcomplex.models.computer_science.evolutionary_computation import robby_ga

from ...conftest import _run


def test_robby_ga_learns_to_collect_cans():
    """Test that the genetic algorithm turns wall-bumping random strategies into ones that score near zero or above."""
    trajectory = _run(robby_ga(), 20)
    first, last = (trajectory.agents('strategies', frame).values['score'] for frame in (0, 20))
    assert first.mean() < -100
    assert last.mean() > first.mean() + 150
    assert last.max() > 0
