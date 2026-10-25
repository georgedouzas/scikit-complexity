"""Test reservoir surrogates: next-generation reservoirs and echo state networks."""

import numpy as np
import pytest

from skcomplex.core.base import AnalysisError
from skcomplex.core.space import DiscreteTime
from skcomplex.engine.analysis import BaseSurrogate, ReservoirSurrogate, Series
from skcomplex.engine.compile import validate
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models import create

LYAPUNOV_TIME = 1 / 0.906
STEP = 0.025
TRANSIENT = 5.0


def _lorenz_samples():
    """Return samples of the Lorenz attractor every 0.025 for 42 Lyapunov times after a transient."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-9, atol=1e-9)
    until = TRANSIENT + 42 * LYAPUNOV_TIME
    trajectory = Simulation(create('lorenz'), solver=solver, dt=STEP).run(until=until).trajectory_
    values = np.column_stack([trajectory[f'lorenz.{name}'] for name in 'xyz'])
    first = round(TRANSIENT / STEP)
    return trajectory.times[first:], values[first:]


def _sine_series():
    """Return a series of two phases of a sine wave sampled at unit steps."""
    times = np.arange(600.0)
    return Series(times, np.column_stack([np.sin(0.1 * times), np.cos(0.1 * times)]), ['x', 'y'])


def test_reservoir_surrogate_forecasts_lorenz_for_a_lyapunov_time():
    """Test that a next-generation reservoir fitted to 40 Lyapunov times forecasts within 10% for one of them."""
    times, values = _lorenz_samples()
    trained = round(40 * LYAPUNOV_TIME / STEP)
    model = ReservoirSurrogate().fit(Series(times[: trained + 1], values[: trained + 1], ['x', 'y', 'z']))
    horizon = round(LYAPUNOV_TIME / STEP)
    forecast = model.run(horizon)
    errors = np.linalg.norm(forecast - values[trained + 1 : trained + horizon + 1], axis=1)
    assert errors.max() < 0.1 * np.ptp(values, axis=0).max()


def test_reservoir_surrogate_runs_under_discrete_time():
    """Test that a reservoir surrogate declares discrete time and that each step applies the readout once."""
    model = ReservoirSurrogate().fit(_sine_series())
    assert isinstance(model.system_.time, DiscreteTime)
    assert validate(model.system_).ok
    run = Simulation(model.system_).run(until=20).trajectory_
    assert np.allclose(np.column_stack([run['surrogate.x'], run['surrogate.y']])[1:], model.run(20))


def test_reservoir_surrogate_holds_memory_as_states():
    """Test that the states of the surrogate are the variables and then the window of the reservoir."""
    model = ReservoirSurrogate(delay=3).fit(_sine_series())
    assert model.states_ == ['x', 'y', *(f'memory_{index}' for index in range(6))]
    assert model.training_error_ < 1e-3
    assert model.validation_error_ < 1e-3


def test_reservoir_surrogate_echo_state_network_forecasts_sine():
    """Test that an echo state network continues a sine wave."""
    model = ReservoirSurrogate(kind='echo_state', units=100, validation=0.0).fit(_sine_series())
    assert len(model.states_) == 102
    expected = np.sin(0.1 * np.arange(600.0, 619.0))
    assert np.allclose(model.run(19)[:, 0], expected, atol=0.05)


def test_reservoir_surrogate_rejects_unknown_kind():
    """Test that an unknown reservoir kind raises an error."""
    surrogate = ReservoirSurrogate(kind='liquid')
    with pytest.raises(AnalysisError, match='unknown'):
        surrogate.check()


def test_reservoir_surrogate_rejects_series_shorter_than_warmup():
    """Test that series with no sample after the warmup raise an error."""
    with pytest.raises(AnalysisError, match='warmup'):
        ReservoirSurrogate(warmup=600).fit(_sine_series())


def test_reservoir_surrogate_run_matches_simulated_system():
    """Test that the fast forecast of a reservoir equals the simulation of its system that the base runs."""
    model = ReservoirSurrogate().fit(_sine_series())
    assert np.allclose(model.run(15), BaseSurrogate._run(model, 15))
