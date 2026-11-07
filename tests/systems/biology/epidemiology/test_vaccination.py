"""Test the `_vaccination` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np
from sklearn.base import clone

from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.systems.biology.epidemiology import SIRVaccinationSystem


def test_sir_vaccination_lowers_the_peak():
    """Test that the optimally vaccinated epidemic peaks lower than the epidemic without vaccination."""
    controlled = simulate(SIRVaccinationSystem())
    uncontrolled = simulate(SIRVaccinationSystem(a=0.0))
    assert np.all(uncontrolled.values('population.u') == 0.0)
    assert controlled.values('population.I').max() < 0.5 * uncontrolled.values('population.I').max()
    assert controlled.values('population.u')[0] > controlled.values('population.u')[-1]


def test_sir_vaccination_records_its_running_cost():
    """Test that the observable `cost` records the running cost of the objective of the default solver."""
    solver = clone(SIRVaccinationSystem.solver).set_params(nodes=51)
    system = SIRVaccinationSystem().prepare().system_
    simulation = (
        Simulation(system, solver=solver, observables=SIRVaccinationSystem.observables).prepare().run(until=5.0)
    )
    assert np.all(simulation.trajectory_.values('cost') > 0)
    assert simulation.stepper_.solution_.cost > 0
