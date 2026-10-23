"""Test the `_vaccination` module of `skcomplex.models.biology.epidemiology`."""

import numpy as np

from skcomplex.core.formalisms.control import Pontryagin
from skcomplex.models import run


def test_sir_vaccination_lowers_the_peak():
    """Test that the optimally vaccinated epidemic peaks lower than the epidemic without vaccination."""
    controlled = run('sir_vaccination').trajectory_
    uncontrolled = run('sir_vaccination', a=0.0).trajectory_
    assert np.all(uncontrolled['population.u'] == 0.0)
    assert controlled['population.I'].max() < 0.5 * uncontrolled['population.I'].max()
    assert controlled['population.u'][0] > controlled['population.u'][-1]


def test_optimal_control_models_record_their_costs():
    """Test that the running cost of the objective is recorded with the solution's cost."""
    simulation = run('sir_vaccination', settings={'until': 5.0, 'solver': Pontryagin(nodes=51)})
    assert np.all(simulation.trajectory_['population.cost'] > 0)
    assert simulation.stepper_.solution_.cost > 0
