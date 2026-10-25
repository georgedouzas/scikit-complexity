"""Test sparse symbolic surrogates fitted by sparse identification of nonlinear dynamics."""

import numpy as np

from skcomplex.core.space import ContinuousTime
from skcomplex.core.system import System
from skcomplex.engine.analysis import Series, SINDySurrogate
from skcomplex.engine.compile import compile_system, validate
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver, resolve_solver
from skcomplex.models import create

LORENZ_TERMS = {
    (0, 'x'): -10.0,
    (0, 'y'): 10.0,
    (1, 'x'): 28.0,
    (1, 'y'): -1.0,
    (1, 'x z'): -1.0,
    (2, 'z'): -8 / 3,
    (2, 'x y'): 1.0,
}


def _lorenz_trajectory():
    """Return a noise-free trajectory of the Lorenz model, sampled every 0.002 for 10 time units."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    return Simulation(create('lorenz'), solver=solver, dt=0.002).run(until=10.0).trajectory_


def test_sindy_surrogate_rediscovers_lorenz():
    """Test that SINDy recovers the seven terms of the Lorenz system with coefficients within 1%."""
    model = SINDySurrogate(degree=2).prepare(_lorenz_trajectory())
    found = {
        (row, feature): value
        for row, values in enumerate(model.coefficients_)
        for feature, value in zip(model.features_, values, strict=True)
        if value
    }
    assert set(found) == set(LORENZ_TERMS)
    assert all(abs(found[term] / value - 1) < 0.01 for term, value in LORENZ_TERMS.items())


def test_sindy_surrogate_surrogate_validates_with_ode_solver():
    """Test that a SINDy surrogate of Lorenz validates, declares continuous time, and gets an ODE solver."""
    model = SINDySurrogate().prepare(_lorenz_trajectory())
    report = validate(model.system_)
    assert not report.errors
    assert isinstance(model.system_.time, ContinuousTime)
    assert isinstance(resolve_solver(compile_system(model.system_), 'auto'), ScipyODESolver)


def test_sindy_surrogate_surrogate_follows_trajectory():
    """Test that the surrogate run from the first sample follows the trajectory for a short time."""
    trajectory = _lorenz_trajectory()
    model = SINDySurrogate().prepare(trajectory)
    run = Simulation(model.system_, dt=0.002).run(until=0.5).trajectory_
    assert np.allclose(run['surrogate.x'], trajectory['lorenz.x'][:251], atol=0.05)
    assert model.training_error_ < 1e-3


def test_sindy_surrogate_fits_series_of_measured_data():
    """Test that SINDy fits a series of measured data with a degree-one library."""
    times = np.arange(0.0, 2.0, 0.001)
    model = SINDySurrogate(degree=1).prepare(Series(times, np.exp(-2 * times)[:, None], ['decay.n']))
    assert model.features_ == ['1', 'n']
    assert model.coefficients_.round(3).tolist() == [[0.0, -2.0]]


def test_sindy_surrogate_run_continues_lotka_volterra():
    """Test that a SINDy surrogate of a Lotka-Volterra run forecasts 50 steps that continue the run."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    trajectory = Simulation(create('lotka_volterra'), solver=solver, dt=0.01).run(until=20.0).trajectory_
    names = ['prey.n', 'predators.n']
    values = np.column_stack([trajectory[name] for name in names])
    model = SINDySurrogate(degree=2, threshold=0.01).prepare(Series(trajectory.times[:1001], values[:1001], names))
    assert isinstance(model.system_, System)
    assert isinstance(model.training_error_, float)
    forecast = model.run(50)
    assert forecast.shape == (50, 2)
    assert np.allclose(forecast, values[1001:1051], rtol=0.01)
