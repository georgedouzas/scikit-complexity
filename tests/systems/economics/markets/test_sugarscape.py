"""Test the `_sugarscape` module of `skcomplex.systems.economics.markets`."""

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.markets import SugarscapeSystem


def test_sugarscape_carrying_capacity_and_inequality():
    """The population falls to a stable carrying capacity and the wealth becomes unequal."""
    system = SugarscapeSystem(30, 30, n_foragers=200).prepare().system_
    trajectory = simulate(
        system,
        until=80,
        random_state=0,
        observables=SugarscapeSystem().get_run_settings()['observables'],
    )
    population, inequality = trajectory.get_values('population'), trajectory.get_values('gini')
    initial = distribution(trajectory.get_agents('foragers', 0).values['sugar'], kind='gini')
    assert population[-1] < 200
    assert abs(population[-1] - population[-20]) <= 0.05 * population[-1]
    assert inequality[-1] > 0.3
    assert inequality[-1] > initial.gini + 0.05


def test_landscape_rises_to_four_at_the_peaks_and_vanishes_in_the_corners():
    """Test that the capacity rings reach 4 near the peaks and 0 in the far corners."""
    capacity = SugarscapeSystem._get_landscape(50, 50)
    assert (float(capacity.max()), float(capacity[0, 49])) == (4.0, 0.0)
