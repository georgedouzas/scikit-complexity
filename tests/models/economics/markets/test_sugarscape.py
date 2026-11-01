"""Test the `_sugarscape` module of `skcomplex.models.economics.markets`."""

from skcomplex.analysis import DistributionAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.economics.markets import sugarscape


def test_sugarscape_carrying_capacity_and_inequality():
    """The population falls to a stable carrying capacity and the wealth becomes unequal."""
    system = sugarscape(30, 30, n_foragers=200)
    trajectory = (
        Simulation(system, random_state=0, observables=describe('sugarscape').observables_of(system))
        .check()
        .prepare()
        .run(until=80)
        .trajectory_
    )
    population, inequality = trajectory.values('population'), trajectory.values('gini')
    initial = DistributionAnalysis('gini').check().prepare(trajectory.agents('foragers', 0).values['sugar']).run()
    assert population[-1] < 200
    assert abs(population[-1] - population[-20]) <= 0.05 * population[-1]
    assert inequality[-1] > 0.3
    assert inequality[-1] > initial.gini_ + 0.05
