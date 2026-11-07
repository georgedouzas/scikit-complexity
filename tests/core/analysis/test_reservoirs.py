"""Test the reservoir analysis: next-generation reservoirs and echo state networks."""

import numpy as np
import pytest

from skcomplex.core.analysis import reservoir
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_system

LYAPUNOV_TIME = 1 / 0.906
STEP = 0.025
TRANSIENT = 5.0


def _lorenz_samples():
    """Return samples of the Lorenz attractor every 0.025 for 42 Lyapunov times after a transient."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-9, atol=1e-9)
    until = TRANSIENT + 42 * LYAPUNOV_TIME
    trajectory = Simulation(create_system('lorenz'), solver=solver, dt=STEP).prepare().run(until=until).trajectory_
    values = np.column_stack([trajectory.values(f'lorenz.{name}') for name in 'xyz'])
    first = round(TRANSIENT / STEP)
    return values[first:]


def _sine_values():
    """Return two phases of a sine wave sampled at unit steps, one column per phase."""
    times = np.arange(600.0)
    return np.column_stack([np.sin(0.1 * times), np.cos(0.1 * times)])


def _fit(source, **settings):
    """Return the reservoir analysis fitted to a source of the variables `x` and `y` by default."""
    return reservoir(source, **{'variables': ['x', 'y'], **settings})


def test_reservoir_analysis_forecasts_lorenz_for_a_lyapunov_time():
    """Test that a next-generation reservoir fitted to 40 Lyapunov times forecasts within 10% for one of them."""
    values = _lorenz_samples()
    trained = round(40 * LYAPUNOV_TIME / STEP)
    horizon = round(LYAPUNOV_TIME / STEP)
    system = _fit(values[: trained + 1], variables=['x', 'y', 'z'], horizon=horizon, dt=STEP)
    forecast = system.forecast
    errors = np.linalg.norm(forecast - values[trained + 1 : trained + horizon + 1], axis=1)
    assert errors.max() < 0.1 * np.ptp(values, axis=0).max()


def test_reservoir_analysis_runs_under_discrete_time():
    """Test that a reservoir surrogate declares discrete time and that each step applies the readout once."""
    system = _fit(_sine_values(), horizon=20)
    assert isinstance(system.system.time, DiscreteTime)
    assert PassCompiler().prepare(system.system).report_.ok
    run = Simulation(system.system).prepare().run(until=20).trajectory_
    assert np.allclose(np.column_stack([run.values('surrogate.x'), run.values('surrogate.y')])[1:], system.forecast)


def test_reservoir_analysis_holds_memory_as_states():
    """Test that the states of the surrogate are the variables and then the window of the reservoir."""
    system = _fit(_sine_values(), delay=3)
    assert system.states == ['x', 'y', *(f'memory_{index}' for index in range(6))]
    assert system.training_error < 1e-3
    assert system.validation_error < 1e-3


def test_reservoir_analysis_echo_state_network_forecasts_sine():
    """Test that an echo state network continues a sine wave."""
    system = _fit(_sine_values(), kind='echo_state', units=100, validation=0.0, horizon=19)
    assert len(system.states) == 102
    expected = np.sin(0.1 * np.arange(600.0, 619.0))
    assert np.allclose(system.forecast[:, 0], expected, atol=0.05)


def test_reservoir_analysis_rejects_unknown_kind():
    """Test that an unknown reservoir kind raises an error."""
    with pytest.raises(AnalysisError, match='unknown'):
        reservoir(np.zeros((10, 1)), kind='liquid')


def test_reservoir_analysis_rejects_series_shorter_than_warmup():
    """Test that series with no sample after the warmup raise an error."""
    with pytest.raises(AnalysisError, match='warmup'):
        _fit(_sine_values(), warmup=600)


def test_next_generation_reads_the_variables_as_they_are_without_standardizing():
    """Test that the setting leaves the mean at 0 and the scale at 1, and the forecast still follows a sine wave."""
    analysis = reservoir(_sine_values(), variables=['u', 'v'], standardize=False, horizon=10)
    assert analysis.step.mean.tolist() == [0.0, 0.0]
    assert analysis.step.scale.tolist() == [1.0, 1.0]
    assert analysis.forecast[-1, 0] == pytest.approx(np.sin(0.1 * 609), abs=0.01)
