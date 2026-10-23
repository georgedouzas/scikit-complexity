"""Test the `_sugarscape` module of `skcomplex.models.economics.markets`."""

from skcomplex.core.runtime import Simulation
from skcomplex.models.economics.markets import sugarscape
from skcomplex.models.economics.markets._sugarscape import gini


def test_sugarscape_carrying_capacity_and_inequality():
    """The population falls to a stable carrying capacity and the wealth becomes unequal."""
    trajectory = Simulation(sugarscape(30, 30, n_foragers=200), random_state=0).run(until=80).trajectory_
    population, inequality = trajectory['census.population'], trajectory['census.gini']
    assert population[-1] < 200
    assert abs(population[-1] - population[-20]) <= 0.05 * population[-1]
    assert inequality[-1] > 0.3
    assert inequality[-1] > gini(trajectory.agents('foragers', 0).values['sugar']) + 0.05
    assert gini([1.0, 1.0, 1.0]) == 0.0
