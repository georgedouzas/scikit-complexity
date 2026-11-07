"""Test the `_sugarscape` module of `skcomplex.systems.economics.markets`."""

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import describe
from skcomplex.systems.economics.markets import sugarscape


def test_sugarscape_carrying_capacity_and_inequality():
    """The population falls to a stable carrying capacity and the wealth becomes unequal."""
    system = sugarscape(30, 30, n_foragers=200)
    trajectory = (
        Simulation(system, random_state=0, observables=describe('sugarscape').observables_of(system))
        .prepare()
        .run(until=80)
        .trajectory_
    )
    population, inequality = trajectory.values('population'), trajectory.values('gini')
    initial = distribution(trajectory.agents('foragers', 0).values['sugar'], kind='gini')
    assert population[-1] < 200
    assert abs(population[-1] - population[-20]) <= 0.05 * population[-1]
    assert inequality[-1] > 0.3
    assert inequality[-1] > initial.gini + 0.05
