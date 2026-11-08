"""Test the sparse identification analysis, which fits sparse symbolic systems to series."""

import numpy as np
import pytest

from skcomplex.core.analysis import sparse_identification
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import LorenzSystem

LORENZ_TERMS = {
    (0, 'x'): -10.0,
    (0, 'y'): 10.0,
    (1, 'x'): 28.0,
    (1, 'y'): -1.0,
    (1, 'x z'): -1.0,
    (2, 'z'): -8 / 3,
    (2, 'x y'): 1.0,
}


def _fit(source, **settings):
    """Return the sparse identification analysis fitted to a source."""
    return sparse_identification(source, **settings)


def _lorenz_trajectory():
    """Return a noise-free trajectory of the Lorenz system, sampled every 0.002 for 10 time units."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    return simulate(LorenzSystem().prepare().system_, until=10.0, solver=solver, dt=0.002)


def test_sparse_identification_rediscovers_lorenz():
    """Test that SINDy recovers the seven terms of the Lorenz system with coefficients within 1%."""
    system = _fit(_lorenz_trajectory(), degree=2)
    found = {
        (row, feature): value
        for row, values in enumerate(system.coefficients)
        for feature, value in zip(system.features, values, strict=True)
        if value
    }
    assert set(found) == set(LORENZ_TERMS)
    assert all(abs(found[term] / value - 1) < 0.01 for term, value in LORENZ_TERMS.items())


def test_sparse_identification_system_validates_with_ode_solver():
    """Test that a SINDy surrogate of Lorenz validates, declares continuous time, and gets an ODE solver."""
    system = _fit(_lorenz_trajectory())
    report = PassCompiler().prepare(system.system).report_
    assert not report.errors
    assert isinstance(system.system.time, ContinuousTime)
    assert DEFAULT_SOLVERS[PassCompiler().prepare(system.system).run().problem_.form] is ScipyODESolver


def test_sparse_identification_system_follows_trajectory():
    """Test that the surrogate run from the first sample follows the trajectory for a short time."""
    trajectory = _lorenz_trajectory()
    system = _fit(trajectory)
    run = simulate(system.system, until=0.5, dt=0.002)
    assert np.allclose(run.values('surrogate.x'), trajectory.values('lorenz.x')[:251], atol=0.05)
    assert system.training_error < 1e-3


def test_sparse_identification_fits_series_of_measured_data():
    """Test that SINDy fits a series of measured data with a degree-one library."""
    times = np.arange(0.0, 2.0, 0.001)
    system = _fit(np.exp(-2 * times), variables=['decay.n'], degree=1, dt=0.001)
    assert system.features == ['1', 'n']
    assert system.coefficients.round(3).tolist() == [[0.0, -2.0]]


def test_sparse_identification_run_continues_lotka_volterra():
    """Test that a SINDy surrogate of a Lotka-Volterra run forecasts 50 steps that continue the run."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    trajectory = simulate(LotkaVolterraSystem().prepare().system_, until=20.0, solver=solver, dt=0.01)
    names = ['prey.n', 'predators.n']
    values = np.column_stack([trajectory.values(name) for name in names])
    system = _fit(values[:1001], variables=names, degree=2, threshold=0.01, horizon=50, dt=0.01)
    assert isinstance(system.system, System)
    assert isinstance(system.training_error, float)
    assert system.forecast.shape == (50, 2)
    assert np.allclose(system.forecast, values[1001:1051], rtol=0.01)


def test_sparse_identification_rejects_a_negative_horizon():
    """Test that a negative horizon raises an error naming it."""
    with pytest.raises(AnalysisError, match='horizon'):
        sparse_identification(np.zeros((10, 1)), horizon=-1)
