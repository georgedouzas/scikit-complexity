"""Test the `_traffic` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np
from sklearn.base import clone

from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import LatticeNeighborhood
from skcomplex.core.systems import System
from skcomplex.systems.engineering.traffic import NagelSchreckenbergInteraction, NagelSchreckenbergSystem


def test_traffic_jams_lower_flow_and_move_backward():
    """Test that dense traffic flows less than near the critical density and its jams move backward."""
    flows, velocities = {}, {}
    for density in (0.15, 0.5):
        trajectory = (
            NagelSchreckenbergSystem(1000, density=density)
            .prepare()
            .system_.simulate(
                until=600,
                dt=1,
                random_state=0,
            )
        )
        velocities[density] = trajectory.get_values('road.velocity')[100:]
        flows[density] = np.where(velocities[density] >= 0, velocities[density], 0).sum(axis=1).mean() / 1000
    assert flows[0.5] < flows[0.15]
    stopped = (velocities[0.5] == 0).astype(float)
    backward = np.mean(stopped[1:] * np.roll(stopped[:-1], -1, axis=1))
    forward = np.mean(stopped[1:] * np.roll(stopped[:-1], 1, axis=1))
    assert backward > 1.5 * forward


def test_traffic_conserves_cars():
    """Test that the number of cars on the ring never changes."""
    trajectory = NagelSchreckenbergSystem(200, density=0.3).prepare().simulate(until=100, dt=1, random_state=1)
    assert np.all((trajectory.get_values('road.velocity') >= 0).sum(axis=1) == 60)


def test_driving_gives_the_velocities_of_the_local_rule_laws():
    """Test that the driving rules as updates with neighbourhoods give the velocities of the `LocalRuleLaw` rules."""
    system = NagelSchreckenbergSystem(300, density=0.3, vmax=4)
    road = system.prepare().system_.elements[0]
    ahead = LatticeNeighborhood(tuple((distance,) for distance in range(5)))
    behind = LatticeNeighborhood(tuple((-distance,) for distance in range(5)))
    rules = [
        LocalRuleInteraction(
            'acceleration',
            target='road',
            state='velocity',
            rule=NagelSchreckenbergInteraction._accelerate,
            neighborhood=ahead,
            stochastic=True,
        ),
        LocalRuleInteraction(
            'movement',
            target='road',
            state='velocity',
            rule=NagelSchreckenbergInteraction._move,
            neighborhood=behind,
        ),
    ]
    reference = System(elements=[clone(road)], interactions=rules).prepare()
    np.testing.assert_array_equal(
        system.simulate(until=100, dt=1, random_state=3).get_values('road.velocity'),
        reference.simulate(until=100, dt=1, random_state=3).get_values('road.velocity'),
    )


def test_driving_shows_its_rules_in_the_equations():
    """Test that the equations name the acceleration and the movement of the cars in the order they run."""
    reads = "neighbors(road.velocity, reduce='each')"
    equations = NagelSchreckenbergSystem(50).prepare().equations_
    assert equations == {'road.velocity[i]': f'accelerate({reads}), then move({reads})'}
