"""Test the `_next_generation` module of `skcomplex.models.computer_science.neural_networks`."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from sklearn.base import clone

from skcomplex.analysis import ReservoirAnalysis
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models import create, run
from skcomplex.models.computer_science.neural_networks import next_generation_reservoir

STEP = 0.025
LYAPUNOV_TIME = 1 / 0.9056
VARIABLES = ['x', 'y', 'z']


def _lorenz_samples(samples=3000, step=STEP, method='RK23', tolerances=(1e-3, 1e-6)):
    """Return samples of the Lorenz flow on its attractor, by the integrator of Gauthier et al. by default."""

    def rate(_, values):
        """Return the Lorenz rates."""
        return [
            10.0 * (values[1] - values[0]),
            values[0] * (28.0 - values[2]) - values[1],
            values[0] * values[1] - 8.0 / 3.0 * values[2],
        ]

    times = step * np.arange(samples + 400)
    rtol, atol = tolerances
    found = solve_ivp(rate, (0.0, times[-1]), [1.0, 1.0, 1.0], t_eval=times, method=method, rtol=rtol, atol=atol)
    return found.y.T[400:]


def _forecast(system, steps, names=VARIABLES):
    """Return the variables of a run of the system, one row per step after the first."""
    trajectory = Simulation(system).check().prepare().run(until=steps).trajectory_
    return np.column_stack([trajectory.values(f'reservoir.{name}') for name in names])[1:]


def test_next_generation_reservoir_forecasts_lorenz():
    """Test that the forecast stays within 10% of the attractor size for at least one Lyapunov time."""
    values = _lorenz_samples()
    trained = 402
    horizon = round(LYAPUNOV_TIME / STEP)
    system = next_generation_reservoir(values[: trained + 1], step=STEP)
    errors = np.linalg.norm(_forecast(system, horizon) - values[trained + 1 : trained + horizon + 1], axis=1)
    assert errors.max() < 0.1 * np.ptp(values, axis=0).max()


def test_next_generation_reservoir_agrees_with_analysis():
    """Test that the forecast of the system agrees with the forecast of the same analysis within 1e-6."""
    values = _lorenz_samples()[:403]
    horizon = round(LYAPUNOV_TIME / STEP)
    analysis = ReservoirAnalysis(
        VARIABLES,
        delay=2,
        order=2,
        ridge=2.5e-6,
        standardize=False,
        validation=0.0,
        horizon=horizon,
        dt=STEP,
    )
    analysis.check().prepare(values).run()
    forecast = _forecast(next_generation_reservoir(analysis=analysis), horizon)
    assert np.abs(forecast - analysis.forecast_).max() < 1e-6 * np.ptp(values, axis=0).max()


def test_next_generation_reservoir_agrees_with_analysis_for_longer_delays():
    """Test that the monomials of the readout stay in the order of the fitted node for three samples of two series."""
    times = 0.1 * np.arange(400.0)
    values = np.column_stack([np.sin(times), np.cos(1.3 * times)])
    analysis = ReservoirAnalysis(['u', 'w'], delay=3, order=3, ridge=1e-6, validation=0.0, horizon=20)
    analysis.check().prepare(values).run()
    forecast = _forecast(next_generation_reservoir(analysis=analysis), 20, ['u', 'w'])
    assert np.abs(forecast - analysis.forecast_).max() < 1e-6


def test_next_generation_reservoir_needs_standardized_features_for_an_accurate_series():
    """Test that an accurate series diverges under the small ridge on raw features and forecasts when standardized."""
    values = _lorenz_samples(method='DOP853', tolerances=(1e-10, 1e-10))
    trained = 402
    horizon = round(LYAPUNOV_TIME / STEP)
    truth = values[trained + 1 : trained + horizon + 1]
    limit = 0.1 * np.ptp(values, axis=0).max()
    standardized = next_generation_reservoir(values[: trained + 1], standardize=True)
    assert np.linalg.norm(_forecast(standardized, horizon) - truth, axis=1).max() < limit
    raw = ReservoirAnalysis(VARIABLES, ridge=2.5e-6, standardize=False, validation=0.0, horizon=horizon, dt=STEP)
    forecast = raw.check().prepare(values[: trained + 1]).run().forecast_
    assert not np.linalg.norm(forecast - truth, axis=1).max() < limit


def test_next_generation_reservoir_keeps_lags_of_every_variable():
    """Test that the states are the variables and `delay - 1` lags of each, starting at the end of the series."""
    values = _lorenz_samples()[:400]
    system = next_generation_reservoir(values, delay=3)
    element = system.elements[0]
    assert (element.x, element.x_lag1, element.x_lag2) == pytest.approx(tuple(values[[-1, -2, -3], 0]))
    assert [interaction.state for interaction in system.interactions] == VARIABLES
    assert all(interaction.mode == 'increment' for interaction in system.interactions)
    trajectory = Simulation(system).check().prepare().run(until=2).trajectory_
    assert trajectory.values('reservoir.x_lag1')[1] == pytest.approx(values[-1, 0])
    assert trajectory.values('reservoir.x_lag2')[2] == pytest.approx(values[-1, 0])


def test_next_generation_reservoir_fits_the_lorenz_flow_by_default():
    """Test that the default model fits the flow it builds and keeps its forecast on the attractor."""
    trajectory = run('next_generation_reservoir').trajectory_
    assert np.ptp(trajectory.values('reservoir.z')) > 20.0
    assert np.abs(trajectory.values('reservoir.x')).max() < 40.0
    assert isinstance(create('next_generation_reservoir', samples=400), System)


def test_next_generation_reservoir_rejects_echo_state_analysis():
    """Test that a readout of another kind of reservoir raises an error naming the kind."""
    values = np.sin(0.1 * np.arange(300.0))[:, None]
    analysis = ReservoirAnalysis(['v'], kind='echo_state', units=20).check().prepare(values).run()
    with pytest.raises(ValidationError, match='echo_state'):
        next_generation_reservoir(analysis=analysis)


def test_next_generation_reservoir_clones():
    """Test that a clone keeps the values of the variables and of their lags."""
    values = np.sin(0.1 * np.arange(200.0))[:, None]
    element = clone(next_generation_reservoir(values, variables=['v'])).elements[0]
    assert (element.v, element.v_lag1) == pytest.approx((values[-1, 0], values[-2, 0]))
