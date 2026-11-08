"""Test the mode decomposition analysis, which fits linear systems to series by DMD and extended DMD."""

import math
from typing import Annotated

import numpy as np
import pytest
import scipy.linalg
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.core.analysis import bind_analysis, mode_decomposition
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

MATRIX = np.asarray([[-0.1, 2.0, 0.0], [-2.0, -0.3, 0.5], [0.0, 0.0, -0.5]])
DRIVEN = np.asarray([[-0.1, 1.0], [-1.0, -0.2]])
DRIVE = np.asarray([0.0, 1.0])
STEP = 0.1


class LinearElement(Element):
    """Linear element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 0.5,
        z: Annotated[object, StateSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z

    def _get_laws(self, s):
        """Return the laws of the linear element."""
        return [
            RateLaw(s.x, float(MATRIX[0, 0]) * s.x + float(MATRIX[0, 1]) * s.y),
            RateLaw(s.y, float(MATRIX[1, 0]) * s.x + float(MATRIX[1, 1]) * s.y + float(MATRIX[1, 2]) * s.z),
            RateLaw(s.z, float(MATRIX[2, 2]) * s.z),
        ]


class DrivenElement(Element):
    """Linear element driven by a bounded input."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 0.0,
        u: Annotated[object, ParameterSpec(adaptive=True, bounds=(-1.0, 1.0))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.u = u

    def _get_laws(self, s):
        """Return the laws of the driven element."""
        return [
            RateLaw(s.x, float(DRIVEN[0, 0]) * s.x + float(DRIVEN[0, 1]) * s.y + float(DRIVE[0]) * s.u),
            RateLaw(s.y, float(DRIVEN[1, 0]) * s.x + float(DRIVEN[1, 1]) * s.y + float(DRIVE[1]) * s.u),
        ]


def _held_input(seed):
    """Return a random piecewise-constant input, held over every 5 samples."""
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, 200)
    return lambda t: float(values[round(t / STEP + 1e-9) // 5])


def _driven_runs():
    """Return runs of the driven element under different random inputs, from different states."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-12)
    return [
        simulate(
            System(elements=[DrivenElement('driven', x=start)]),
            until=20.0,
            solver=solver,
            dt=STEP,
            schedule={'driven.u': _held_input(seed)},
        )
        for seed, start in enumerate((1.0, -2.0, 0.5))
    ]


def _one_sample_map():
    """Return the matrices of the map over one sample of the driven element under a held input."""
    block = np.zeros((3, 3))
    block[:2, :2] = DRIVEN
    block[:2, 2] = DRIVE
    flow = scipy.linalg.expm(block * STEP)
    return flow[:2, :2], flow[:2, 2:]


def _linear_trajectory():
    """Return a trajectory of the linear system `dx/dt = A x`, sampled every 0.05 for 10 time units."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-12)
    return simulate(System(elements=[LinearElement('linear')]), until=10.0, solver=solver, dt=0.05)


NAMES = ['linear.x', 'linear.y', 'linear.z']


def _fit(source, **settings):
    """Return the mode decomposition analysis fitted to a source."""
    return mode_decomposition(source, **settings)


def _linear_values(trajectory, samples):
    """Return the first samples of the states of the linear trajectory, one column per state."""
    return np.column_stack([trajectory.values(name)[:samples] for name in NAMES])


def test_mode_decomposition_recovers_linear_eigenvalues():
    """Test that the continuous eigenvalues of DMD equal those of the system matrix within 1e-6."""
    decomposition = _fit(_linear_trajectory())
    expected = np.sort_complex(np.linalg.eigvals(MATRIX))
    assert np.abs(np.sort_complex(decomposition.eigenvalues) - expected).max() < 1e-6
    assert np.allclose(decomposition.generator, MATRIX, atol=1e-6)


def test_mode_decomposition_system_validates_under_continuous_time():
    """Test that the surrogate passes validation, declares continuous time, and runs with an ODE solver."""
    trajectory = _linear_trajectory()
    decomposition = _fit(trajectory)
    assert PassCompiler().prepare(decomposition.system).report_.ok
    assert isinstance(decomposition.system.time, ContinuousTime)
    assert DEFAULT_SOLVERS[PassCompiler().prepare(decomposition.system).run().problem_.form] is ScipyODESolver
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    run = simulate(decomposition.system, until=10.0, solver=solver, dt=0.05)
    assert np.allclose(run.values('surrogate.y'), trajectory.values('linear.y'), atol=1e-6)


def test_mode_decomposition_run_continues_trajectory():
    """Test that running the modes continues the trajectory after the last sample it was fitted to."""
    trajectory = _linear_trajectory()
    decomposition = _fit(_linear_values(trajectory, 101), variables=NAMES, horizon=50, dt=0.05)
    assert decomposition.labels == ['linear.x', 'linear.y', 'linear.z']
    assert decomposition.states == ['x', 'y', 'z']
    assert decomposition.forecast.shape == (50, 3)
    assert np.allclose(decomposition.forecast[:, 2], trajectory.values('linear.z')[101:151], atol=1e-8)


def test_mode_decomposition_run_matches_simulated_system():
    """Test that the forecast of the modes equals the simulation of the fitted system from the last sample."""
    decomposition = _fit(_linear_values(_linear_trajectory(), 101), variables=NAMES, horizon=20, dt=0.05)
    names = [f'surrogate.{name}' for name in decomposition.states]
    initial_state = dict(zip(names, decomposition.last_state, strict=True))
    run = simulate(decomposition.system, until=1.0, dt=0.05, initial_state=initial_state)
    simulated = np.column_stack([run.values(name) for name in names])[1:21]
    assert np.allclose(decomposition.forecast, simulated, atol=1e-4)


def test_mode_decomposition_binds_without_checking():
    """Test that an invalid validation share binds and fails when the bound analysis is called."""
    surrogate = bind_analysis(mode_decomposition, validation=1.0)
    assert surrogate.keywords['validation'] == 1.0
    with pytest.raises(AnalysisError, match='validation share'):
        surrogate(np.zeros((10, 2)))


def test_mode_decomposition_reports_small_errors_on_linear_data():
    """Test that the training and validation errors of a linear system vanish."""
    decomposition = _fit(_linear_trajectory())
    assert decomposition.training_error < 1e-8
    assert decomposition.validation_error < 1e-8


def test_mode_decomposition_without_validation_has_no_validation_error():
    """Test that fitting every sample leaves the validation error undefined."""
    assert math.isnan(_fit(_linear_trajectory(), validation=0.0).validation_error)


def test_mode_decomposition_rank_truncates_modes():
    """Test that a rank keeps that many modes."""
    decomposition = _fit(_linear_trajectory(), rank=2)
    assert decomposition.modes.shape == (3, 2)
    assert decomposition.eigenvalues.shape == (2,)


def test_mode_decomposition_dictionary_lifts_observables():
    """Test that extended DMD runs on the observables of a dictionary, which become the states of the surrogate."""
    trajectory = _linear_trajectory()
    decomposition = _fit(
        trajectory,
        variables=['linear.x', 'linear.y'],
        dictionary=PolynomialFeatures(2, include_bias=False),
    )
    assert decomposition.labels == ['linear.x', 'linear.y', 'linear.x^2', 'linear.x linear.y', 'linear.y^2']
    assert decomposition.states == ['x', 'y', 'x_2', 'x_y', 'y_2']
    assert PassCompiler().prepare(decomposition.system).report_.ok


def test_mode_decomposition_fits_several_series():
    """Test that DMD fits the pairs of several runs of the same system, given as a list or a 3-D array."""
    times = np.arange(0.0, 5.0, 0.1)
    runs = [(start * np.exp(-0.5 * times))[:, None] for start in (1.0, -2.0)]
    decomposition = _fit(runs, variables=['x'], dt=0.1)
    assert _fit(np.stack(runs), variables=['x'], dt=0.1).eigenvalues.tolist() == decomposition.eigenvalues.tolist()
    assert decomposition.eigenvalues.real.round(8).tolist() == [-0.5]
    assert decomposition.system.elements[0].get_value('x') == 1.0


def test_mode_decomposition_rejects_series_of_other_variables():
    """Test that series of different variables raise an error."""
    trajectory = _linear_trajectory()
    with pytest.raises(AnalysisError, match='same variables'):
        mode_decomposition([trajectory, np.zeros((10, 3))])


def test_mode_decomposition_rejects_no_series():
    """Test that an empty sequence of series raises an error."""
    with pytest.raises(AnalysisError, match='No series'):
        mode_decomposition([])


def test_mode_decomposition_rejects_validation_share_of_one():
    """Test that a validation share outside `[0, 1)` raises an error."""
    with pytest.raises(AnalysisError, match='validation share'):
        mode_decomposition(np.zeros((10, 2)), validation=1.0)


def test_mode_decomposition_recovers_map_with_input():
    """Test that the operator and the input matrix equal the one-sample map of a linear system within 1e-6."""
    decomposition = _fit(_driven_runs(), variables=['driven.x', 'driven.y'], inputs=['driven.u'], validation=0.0)
    operator, input_matrix = _one_sample_map()
    assert decomposition.states == ['x', 'y']
    assert decomposition.inputs == ['u']
    assert np.abs(decomposition.operator - operator).max() < 1e-6
    assert np.abs(decomposition.input_matrix - input_matrix).max() < 1e-6


def test_mode_decomposition_system_with_input_follows_schedule():
    """Test that the system records a scheduled input at every frame and that its states follow the fitted map."""
    decomposition = _fit(_driven_runs(), inputs=['driven.u'], variables=['driven.x', 'driven.y'], validation=0.0)
    schedule = _held_input(7)
    solver = ScipyODESolver(method='DOP853', rtol=1e-11, atol=1e-11)
    run = simulate(decomposition.system, until=5.0, solver=solver, dt=STEP, schedule={'surrogate.u': schedule})
    recorded = run.values('surrogate.u')
    assert np.allclose(recorded, [schedule(time) for time in run.times])
    states = np.column_stack([run.values('surrogate.x'), run.values('surrogate.y')])
    predicted = states[:-1] @ decomposition.operator.T + recorded[:-1, None] @ decomposition.input_matrix.T
    assert np.abs(states[1:] - predicted).max() < 1e-6


def test_mode_decomposition_bounds_inputs_by_samples():
    """Test that the input of the fitted system is an adaptive parameter bounded by its range in the samples."""
    decomposition = _fit(_driven_runs(), inputs=['driven.u'], variables=['driven.x', 'driven.y'])
    specification = PassCompiler().prepare(decomposition.system).run().problem_.adaptive_parameters['surrogate.u']
    joined = np.concatenate([run.values('driven.u') for run in _driven_runs()])
    assert specification.adaptive
    assert specification.bounds == (joined.min(), joined.max())


def test_mode_decomposition_forecast_holds_last_input():
    """Test that the forecast iterates the fitted map with the input held at its last value."""
    decomposition = _fit(
        _driven_runs(),
        inputs=['driven.u'],
        variables=['driven.x', 'driven.y'],
        validation=0.1,
        horizon=2,
    )
    last = _driven_runs()[-1].values('driven.u')[-1]
    first = decomposition.operator @ decomposition.last_state + decomposition.input_matrix[:, 0] * last
    assert np.allclose(decomposition.forecast[0], first)
    assert decomposition.validation_error < 1e-6


def test_mode_decomposition_input_with_dictionary_excludes_input():
    """Test that the dictionary computes the observables of the variables that are not inputs."""
    decomposition = _fit(
        _driven_runs(),
        variables=['driven.x', 'driven.y'],
        inputs=['driven.u'],
        dictionary=PolynomialFeatures(2, include_bias=False),
    )
    assert decomposition.states == ['x', 'y', 'x_2', 'x_y', 'y_2']
    assert decomposition.input_matrix.shape == (5, 1)
    assert PassCompiler().prepare(decomposition.system).report_.ok


def test_mode_decomposition_input_rank_truncates_modes():
    """Test that a rank keeps that many modes of the map with inputs."""
    decomposition = _fit(_driven_runs(), rank=1, inputs=['driven.u'], variables=['driven.x', 'driven.y'])
    assert decomposition.modes.shape == (2, 1)


def test_mode_decomposition_rejects_input_outside_variables():
    """Test that an input that the series lacks raises an error naming it."""
    trajectory = _linear_trajectory()
    with pytest.raises(AnalysisError, match=r'linear\.q'):
        mode_decomposition(trajectory, inputs=['linear.q'])


def test_mode_decomposition_input_name_differs_from_states():
    """Test that an input with the last part of the name of a state gets a name of its own."""
    values = np.random.default_rng(0).normal(size=(50, 2))
    decomposition = _fit(values, variables=['a.x', 'b.x'], inputs=['b.x'])
    assert decomposition.inputs == ['x_input']
