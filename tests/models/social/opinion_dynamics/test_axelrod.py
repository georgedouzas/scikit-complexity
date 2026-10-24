"""Test the `_axelrod` module of `skcomplex.models.social.opinion_dynamics`."""

from skcomplex.engine.runtime import Simulation
from skcomplex.models.social.opinion_dynamics import axelrod, cultural_traits, largest_region


def test_axelrod_order_disorder_transition():
    """For few traits one culture spans most of the lattice, and for many traits small regions freeze."""
    regions = {}
    for traits in (2, 50):
        system = axelrod(8, 8, features=3, traits=traits, random_state=0)
        trajectory = Simulation(system, random_state=0).run(until=300).trajectory_
        regions[traits] = trajectory['census.largest_region'][-1]
    assert regions[2] > 0.5
    assert regions[50] < 0.1


def test_axelrod_helpers():
    """Culture codes decode into base-`q` digits, and regions are connected cells of one culture."""
    assert cultural_traits(2 + 3 * 5, 2, 5).tolist() == [2, 3]
    assert largest_region([(0, 0), (1, 0), (0, 1), (1, 1)], [4, 7, 7, 4], 2, 2) == 0.25
