"""Test the `_dynamic_fba` module of `skcomplex.models.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.errors import ValidationError
from skcomplex.models import run
from skcomplex.models.biology.systems_biology import dynamic_fba

KM = 0.015


def test_batch_culture_grows_in_one_phase_until_the_glucose_ends():
    """Test the falling glucose, the growth rate, and the plateau of the batch culture."""
    trajectory = run('dynamic_fba').trajectory_
    biomass = trajectory.values('culture/biomass.x')
    glucose = trajectory.values('culture/glucose.c')
    assert np.all(np.diff(glucose) <= 1e-12)
    assert glucose[-1] < 1e-3
    plenty = glucose > 10 * KM
    slope = np.polyfit(trajectory.times[plenty], np.log(biomass[plenty]), 1)[0]
    assert slope == pytest.approx(0.874, rel=0.01)
    assert biomass[-1] == pytest.approx(0.01 + 0.0874 * 10, rel=0.02)
    assert np.all(np.diff(biomass) >= -1e-12)


def test_batch_culture_records_a_zero_exchange_after_depletion():
    """Test that the glucose exchange is recorded at every frame and is zero after depletion."""
    trajectory = run('dynamic_fba').trajectory_
    exchange = trajectory.values('fba/cell.v_EX_glc__D_e')
    assert exchange.size == trajectory.times.size
    assert exchange[0] == pytest.approx(-10, rel=0.01)
    assert exchange[-1] == pytest.approx(0.0, abs=1e-9)


def test_static_optimisation_approaches_the_direct_approach():
    """Test that the biomass at 6 h approaches the direct approach, with an error that falls at least linearly."""
    direct = run('dynamic_fba', approach='direct', settings={'until': 6.0}).trajectory_
    reference = float(direct.values('biomass.x')[-1])
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(sync_dt=interval, solvers={'culture': RK4Solver(dt=0.005)})
        simulation = Simulation(dynamic_fba(interval=interval), solver=solver, dt=0.1).check().prepare().run(6.0)
        errors.append(abs(float(simulation.trajectory_.values('culture/biomass.x')[-1]) - reference))
    assert errors[1] <= errors[0] * 0.75 + 1e-9
    assert errors[2] <= errors[1] * 0.75 + 1e-9


def test_dynamic_fba_rejects_an_unknown_approach():
    """Test that an unknown approach raises an error that names the approaches."""
    with pytest.raises(ValidationError, match='static, direct'):
        dynamic_fba('mixed')
