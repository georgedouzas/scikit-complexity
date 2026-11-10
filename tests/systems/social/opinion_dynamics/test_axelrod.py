"""Test the `_axelrod` module of `skcomplex.systems.social.opinion_dynamics`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.social.opinion_dynamics import (
    AxelrodSystem,
    cultural_traits,
    culture_observables,
    largest_region,
)


def test_axelrod_order_disorder_transition():
    """For few traits one culture spans most of the lattice, and for many traits small regions freeze."""
    regions = {}
    for traits in (2, 50):
        system = AxelrodSystem(8, 8, features=3, traits=traits, random_state=0).prepare().system_
        trajectory = simulate(system, until=300, random_state=0, observables=culture_observables(8, 8))
        regions[traits] = trajectory.get_values('largest_region')[-1]
    assert regions[2] > 0.5
    assert regions[50] < 0.1


def test_axelrod_helpers():
    """Culture codes decode into base-`q` digits, and regions are connected cells of one culture."""
    assert cultural_traits(2 + 3 * 5, 2, 5).tolist() == [2, 3]
    assert largest_region([(0, 0), (1, 0), (0, 1), (1, 1)], [4, 7, 7, 4], 2, 2) == 0.25
