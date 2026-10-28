"""Test the reservoir analysis: next-generation reservoirs and echo state networks."""

import numpy as np
import pytest

from skcomplex.analysis import ReservoirAnalysis
from skcomplex.core.spaces import DiscreteTime
from skcomplex.engine.compile import PassCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.errors import AnalysisError
from skcomplex.models import create

LYAPUNOV_TIME = 1 / 0.906
STEP = 0.025
TRANSIENT = 5.0


def _lorenz_samples():
    """Return samples of the Lorenz attractor every 0.025 for 42 Lyapunov times after a transient."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-9, atol=1e-9)
    until = TRANSIENT + 42 * LYAPUNOV_TIME
    trajectory = Simulation(create('lorenz'), solver=solver, dt=STEP).check().run(until=until).trajectory_
    values = np.column_stack([trajectory.values(f'lorenz.{name}') for name in 'xyz'])
    first = round(TRANSIENT / STEP)
    return values[first:]


def _sine_values():
    """Return two phases of a sine wave sampled at unit steps, one column per phase."""
    times = np.arange(600.0)
    return np.column_stack([np.sin(0.1 * times), np.cos(0.1 * times)])


def _fit(source, **settings):
    """Return the reservoir analysis fitted to a source of the variables `x` and `y` by default."""
    return ReservoirAnalysis(**{'variables': ['x', 'y'], **settings}).check().prepare(source).run()


def test_reservoir_analysis_forecasts_lorenz_for_a_lyapunov_time():
    """Test that a next-generation reservoir fitted to 40 Lyapunov times forecasts within 10% for one of them."""
    values = _lorenz_samples()
    trained = round(40 * LYAPUNOV_TIME / STEP)
    horizon = round(LYAPUNOV_TIME / STEP)
    model = _fit(values[: trained + 1], variables=['x', 'y', 'z'], horizon=horizon, dt=STEP)
    forecast = model.forecast_
    errors = np.linalg.norm(forecast - values[trained + 1 : trained + horizon + 1], axis=1)
    assert errors.max() < 0.1 * np.ptp(values, axis=0).max()


def test_reservoir_analysis_runs_under_discrete_time():
    """Test that a reservoir surrogate declares discrete time and that each step applies the readout once."""
    model = _fit(_sine_values(), horizon=20)
    assert isinstance(model.system_.time, DiscreteTime)
    assert PassCompiler().check().prepare(model.system_).report_.ok
    run = Simulation(model.system_).check().run(until=20).trajectory_
    assert np.allclose(np.column_stack([run.values('surrogate.x'), run.values('surrogate.y')])[1:], model.forecast_)


def test_reservoir_analysis_holds_memory_as_states():
    """Test that the states of the surrogate are the variables and then the window of the reservoir."""
    model = _fit(_sine_values(), delay=3)
    assert model.states_ == ['x', 'y', *(f'memory_{index}' for index in range(6))]
    assert model.training_error_ < 1e-3
    assert model.validation_error_ < 1e-3


def test_reservoir_analysis_echo_state_network_forecasts_sine():
    """Test that an echo state network continues a sine wave."""
    model = _fit(_sine_values(), kind='echo_state', units=100, validation=0.0, horizon=19)
    assert len(model.states_) == 102
    expected = np.sin(0.1 * np.arange(600.0, 619.0))
    assert np.allclose(model.forecast_[:, 0], expected, atol=0.05)


def test_reservoir_analysis_rejects_unknown_kind():
    """Test that an unknown reservoir kind raises an error."""
    surrogate = ReservoirAnalysis(kind='liquid')
    with pytest.raises(AnalysisError, match='unknown'):
        surrogate.check()


def test_reservoir_analysis_rejects_series_shorter_than_warmup():
    """Test that series with no sample after the warmup raise an error."""
    with pytest.raises(AnalysisError, match='warmup'):
        _fit(_sine_values(), warmup=600)


