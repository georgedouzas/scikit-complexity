"""Test sparse symbolic surrogates fitted by sparse identification of nonlinear dynamics."""

import numpy as np

from skcomplex.core.analysis import Series, fit_sindy
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE, resolve_solver
from skcomplex.core.space import ContinuousTime
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
    solver = ScipyODE(method='DOP853', rtol=1e-10, atol=1e-10)
    return Simulation(create('lorenz'), solver=solver, dt=0.002).run(until=10.0).trajectory_


def test_fit_sindy_rediscovers_lorenz():
    """Test that SINDy recovers the seven terms of the Lorenz system with coefficients within 1%."""
    model = fit_sindy(_lorenz_trajectory(), degree=2)
    found = {
        (row, feature): value
        for row, values in enumerate(model.coefficients)
        for feature, value in zip(model.features, values, strict=True)
        if value
    }
    assert set(found) == set(LORENZ_TERMS)
    assert all(abs(found[term] / value - 1) < 0.01 for term, value in LORENZ_TERMS.items())


def test_fit_sindy_surrogate_validates_with_ode_solver():
    """Test that a SINDy surrogate of Lorenz validates, declares continuous time, and gets an ODE solver."""
    model = fit_sindy(_lorenz_trajectory())
    report = validate(model.system)
    assert not report.errors
    assert isinstance(model.system.time, ContinuousTime)
    assert isinstance(resolve_solver(compile_system(model.system), 'auto'), ScipyODE)


def test_fit_sindy_surrogate_follows_trajectory():
    """Test that the surrogate run from the first sample follows the trajectory for a short time."""
    trajectory = _lorenz_trajectory()
    model = fit_sindy(trajectory)
    run = Simulation(model.system, dt=0.002).run(until=0.5).trajectory_
    assert np.allclose(run['surrogate.x'], trajectory['lorenz.x'][:251], atol=0.05)
    assert model.training_error < 1e-3


def test_fit_sindy_fits_series_of_measured_data():
    """Test that SINDy fits a series of measured data with a degree-one library."""
    times = np.arange(0.0, 2.0, 0.001)
    model = fit_sindy(Series(times, np.exp(-2 * times)[:, None], ['decay.n']), degree=1)
    assert model.features == ['1', 'n']
    assert model.coefficients.round(3).tolist() == [[0.0, -2.0]]
