"""Test the `_next_generation` module of `skcomplex.systems.computer_science.neural_networks`."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from sklearn.base import clone

from skcomplex.core.analysis import reservoir
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_system
from skcomplex.systems.computer_science.neural_networks import NextGenerationReservoirSystem

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
    trajectory = simulate(system, until=steps)
    return np.column_stack([trajectory.values(f'reservoir.{name}') for name in names])[1:]


def test_next_generation_reservoir_forecasts_lorenz():
    """Test that the forecast stays within 10% of the attractor size for at least one Lyapunov time."""
    values = _lorenz_samples()
    trained = 402
    horizon = round(LYAPUNOV_TIME / STEP)
    system = NextGenerationReservoirSystem(values[: trained + 1], step=STEP)
    errors = np.linalg.norm(_forecast(system, horizon) - values[trained + 1 : trained + horizon + 1], axis=1)
    assert errors.max() < 0.1 * np.ptp(values, axis=0).max()


def test_next_generation_reservoir_agrees_with_analysis():
    """Test that the forecast of the system agrees with the forecast of the same analysis within 1e-6."""
    values = _lorenz_samples()[:403]
    horizon = round(LYAPUNOV_TIME / STEP)
    analysis = reservoir(
        values,
        variables=VARIABLES,
        delay=2,
        order=2,
        ridge=2.5e-6,
        standardize=False,
        validation=0.0,
        horizon=horizon,
        dt=STEP,
    )
    forecast = _forecast(NextGenerationReservoirSystem(result=analysis), horizon)
    assert np.abs(forecast - analysis.forecast).max() < 1e-6 * np.ptp(values, axis=0).max()


def test_next_generation_reservoir_agrees_with_analysis_for_longer_delays():
    """Test that the monomials of the readout stay in the order of the fitted node for three samples of two series."""
    times = 0.1 * np.arange(400.0)
    values = np.column_stack([np.sin(times), np.cos(1.3 * times)])
    analysis = reservoir(values, variables=['u', 'w'], delay=3, order=3, ridge=1e-6, validation=0.0, horizon=20)
    forecast = _forecast(NextGenerationReservoirSystem(result=analysis), 20, ['u', 'w'])
    assert np.abs(forecast - analysis.forecast).max() < 1e-6


def test_next_generation_reservoir_needs_standardized_features_for_an_accurate_series():
    """Test that an accurate series diverges under the small ridge on raw features and forecasts when standardized."""
    values = _lorenz_samples(method='DOP853', tolerances=(1e-10, 1e-10))
    trained = 402
    horizon = round(LYAPUNOV_TIME / STEP)
    truth = values[trained + 1 : trained + horizon + 1]
    limit = 0.1 * np.ptp(values, axis=0).max()
    standardized = NextGenerationReservoirSystem(values[: trained + 1], standardize=True)
    assert np.linalg.norm(_forecast(standardized, horizon) - truth, axis=1).max() < limit
    forecast = reservoir(
        values[: trained + 1],
        variables=VARIABLES,
        ridge=2.5e-6,
        standardize=False,
        validation=0.0,
        horizon=horizon,
        dt=STEP,
    ).forecast
    assert not np.linalg.norm(forecast - truth, axis=1).max() < limit


def test_next_generation_reservoir_keeps_lags_of_every_variable():
    """Test that the states are the variables and `delay - 1` lags of each, starting at the end of the series."""
    values = _lorenz_samples()[:400]
    system = NextGenerationReservoirSystem(values, delay=3).prepare().system_
    element = system.elements[0]
    assert element.get_value('x') == pytest.approx(values[-1, 0])
    assert element.variables['x'].spec.initial == pytest.approx(tuple(values[[-2, -3], 0]))
    assert [interaction.state for interaction in system.interactions] == VARIABLES
    assert all(interaction.mode == 'increment' for interaction in system.interactions)
    trajectory = simulate(system, until=2)
    assert trajectory.values('reservoir.x_lag1')[1] == pytest.approx(values[-1, 0])
    assert trajectory.values('reservoir.x_lag2')[2] == pytest.approx(values[-1, 0])


def test_next_generation_reservoir_fits_the_lorenz_flow_by_default():
    """Test that the default system fits the flow it builds and keeps its forecast on the attractor."""
    trajectory = simulate(NextGenerationReservoirSystem(), random_state=0)
    assert np.ptp(trajectory.values('reservoir.z')) > 20.0
    assert np.abs(trajectory.values('reservoir.x')).max() < 40.0
    assert isinstance(create_system('next_generation_reservoir', samples=400), System)


def test_next_generation_reservoir_rejects_echo_state_analysis():
    """Test that a readout of another kind of reservoir raises an error naming the kind."""
    values = np.sin(0.1 * np.arange(300.0))[:, None]
    analysis = reservoir(values, variables=['v'], kind='echo_state', units=20)
    with pytest.raises(ValidationError, match='echo_state'):
        NextGenerationReservoirSystem(result=analysis).prepare()


def test_next_generation_reservoir_clones():
    """Test that a clone keeps the values of the variables and of their lags."""
    values = np.sin(0.1 * np.arange(200.0))[:, None]
    element = clone(NextGenerationReservoirSystem(values, variables=['v'])).prepare().system_.elements[0]
    assert element.get_value('v') == pytest.approx(values[-1, 0])
    assert element.variables['v'].spec.initial == pytest.approx((values[-2, 0],))


def test_next_generation_reservoir_keeps_its_reference_values():
    """Test a forecast against the values recorded before the element moved to declared states."""
    values = np.sin(0.1 * np.arange(200.0))[:, None]
    system = NextGenerationReservoirSystem(values, variables=['v'])
    trajectory = simulate(system, until=20, dt=1)
    last = [float(np.ravel(trajectory.values(path))[-1]) for path in ('reservoir.v', 'reservoir.v_lag1')]
    assert last == pytest.approx([0.09103560049023797, 0.18999621778271003], rel=1e-9)
