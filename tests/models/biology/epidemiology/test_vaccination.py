"""Test the `_vaccination` module of `skcomplex.models.biology.epidemiology`."""

import numpy as np
from sklearn.base import clone

from skcomplex.models import describe, run


def test_sir_vaccination_lowers_the_peak():
    """Test that the optimally vaccinated epidemic peaks lower than the epidemic without vaccination."""
    controlled = run('sir_vaccination').trajectory_
    uncontrolled = run('sir_vaccination', a=0.0).trajectory_
    assert np.all(uncontrolled.values('population.u') == 0.0)
    assert controlled.values('population.I').max() < 0.5 * uncontrolled.values('population.I').max()
    assert controlled.values('population.u')[0] > controlled.values('population.u')[-1]


def test_sir_vaccination_records_its_running_cost():
    """Test that the observable `cost` records the running cost of the objective of the default solver."""
    solver = clone(describe('sir_vaccination').defaults['solver']).set_params(nodes=51)
    simulation = run('sir_vaccination', settings={'until': 5.0, 'solver': solver})
    assert np.all(simulation.trajectory_.values('cost') > 0)
    assert simulation.stepper_.solution_.cost > 0
