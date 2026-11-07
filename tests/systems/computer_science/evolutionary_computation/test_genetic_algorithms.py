"""Test the `_genetic_algorithms` module of `skcomplex.systems.computer_science.evolutionary_computation`."""

import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.evolutionary_computation import RobbyGASystem

from ..conftest import _run


def test_robby_ga_learns_to_collect_cans():
    """Test that the genetic algorithm turns wall-bumping random strategies into ones that score near zero or above."""
    trajectory = _run(RobbyGASystem(), 20)
    first, last = (trajectory.agents('strategies', frame).values['score'] for frame in (0, 20))
    assert first.mean() < -100
    assert last.mean() > first.mean() + 150
    assert last.max() > 0


def test_robby_ga_by_circumstance_improves_only_under_starvation():
    """Test that Gooding's evolution by circumstance raises the scores with 30% starvation and not without it."""
    means = {}
    for starvation in (0.3, 0.0):
        system = RobbyGASystem(regime='circumstance', starvation=starvation)
        trajectory = _run(system, 40)
        means[starvation] = trajectory.agents('strategies', 40).values['score'].mean()
        ages = trajectory.agents('strategies', 40).values['age']
        assert ages.max() < 7
    assert means[0.3] > -50
    assert means[0.0] < -150


def test_robby_ga_rejects_an_unknown_regime():
    """Test that an unknown regime raises a parameter error."""
    with pytest.raises(ParameterError, match='Unknown regime'):
        RobbyGASystem(regime='lamarckian').prepare()
