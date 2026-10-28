"""Test the mode decomposition analysis, which fits linear systems to series by DMD and extended DMD."""

import math
from typing import Annotated

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.analysis import ModeDecompositionAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.spaces import ContinuousTime
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.errors import AnalysisError
from skcomplex.specs import StateSpec, VisualSpec

MATRIX = np.asarray([[-0.1, 2.0, 0.0], [-2.0, -0.3, 0.5], [0.0, 0.0, -0.5]])


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


def _linear_trajectory():
    """Return a trajectory of the linear system `dx/dt = A x`, sampled every 0.05 for 10 time units."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-12)
    return (
        Simulation(System(elements=[LinearElement('linear')]), solver=solver, dt=0.05)
        .check()
        .run(until=10.0)
        .trajectory_
    )


NAMES = ['linear.x', 'linear.y', 'linear.z']


def _fit(source, **settings):
    """Return the mode decomposition analysis fitted to a source."""
    return ModeDecompositionAnalysis(**settings).check().prepare(source).run()


def _linear_values(trajectory, samples):
    """Return the first samples of the states of the linear trajectory, one column per state."""
    return np.column_stack([trajectory.values(name)[:samples] for name in NAMES])


def test_mode_decomposition_recovers_linear_eigenvalues():
    """Test that the continuous eigenvalues of DMD equal those of the system matrix within 1e-6."""
    decomposition = _fit(_linear_trajectory())
    expected = np.sort_complex(np.linalg.eigvals(MATRIX))
    assert np.abs(np.sort_complex(decomposition.eigenvalues_) - expected).max() < 1e-6
    assert np.allclose(decomposition.generator_, MATRIX, atol=1e-6)


def test_mode_decomposition_system_validates_under_continuous_time():
    """Test that the surrogate passes validation, declares continuous time, and runs with an ODE solver."""
    trajectory = _linear_trajectory()
    decomposition = _fit(trajectory)
    assert PassCompiler().check().prepare(decomposition.system_).report_.ok
    assert isinstance(decomposition.system_.time, ContinuousTime)
    assert DEFAULT_SOLVERS[compile_system(decomposition.system_).form] is ScipyODESolver
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    run = Simulation(decomposition.system_, solver=solver, dt=0.05).check().run(until=10.0).trajectory_
    assert np.allclose(run.values('surrogate.y'), trajectory.values('linear.y'), atol=1e-6)


def test_mode_decomposition_run_continues_trajectory():
    """Test that running the modes continues the trajectory after the last sample it was fitted to."""
    trajectory = _linear_trajectory()
    decomposition = _fit(_linear_values(trajectory, 101), variables=NAMES, horizon=50, dt=0.05)
    assert decomposition.labels_ == ['linear.x', 'linear.y', 'linear.z']
    assert decomposition.states_ == ['x', 'y', 'z']
    assert decomposition.forecast_.shape == (50, 3)
    assert np.allclose(decomposition.forecast_[:, 2], trajectory.values('linear.z')[101:151], atol=1e-8)


def test_mode_decomposition_run_matches_simulated_system():
    """Test that the forecast of the modes equals the simulation of the fitted system from the last sample."""
    decomposition = _fit(_linear_values(_linear_trajectory(), 101), variables=NAMES, horizon=20, dt=0.05)
    names = [f'surrogate.{name}' for name in decomposition.states_]
    initial_state = dict(zip(names, decomposition.last_state_, strict=True))
    run = Simulation(decomposition.system_, dt=0.05, initial_state=initial_state).check().run(until=1.0).trajectory_
    simulated = np.column_stack([run.values(name) for name in names])[1:21]
    assert np.allclose(decomposition.forecast_, simulated, atol=1e-4)


def test_mode_decomposition_builds_without_checking():
    """Test that an invalid validation share builds and fails when checked."""
    surrogate = ModeDecompositionAnalysis(validation=1.0)
    assert surrogate.get_params()['validation'] == 1.0
    with pytest.raises(AnalysisError, match='validation share'):
        surrogate.check()


def test_mode_decomposition_reports_small_errors_on_linear_data():
    """Test that the training and validation errors of a linear system vanish."""
    decomposition = _fit(_linear_trajectory())
    assert decomposition.training_error_ < 1e-8
    assert decomposition.validation_error_ < 1e-8


def test_mode_decomposition_without_validation_has_no_validation_error():
    """Test that fitting every sample leaves the validation error undefined."""
    assert math.isnan(_fit(_linear_trajectory(), validation=0.0).validation_error_)


def test_mode_decomposition_rank_truncates_modes():
    """Test that a rank keeps that many modes."""
    decomposition = _fit(_linear_trajectory(), rank=2)
    assert decomposition.modes_.shape == (3, 2)
    assert decomposition.eigenvalues_.shape == (2,)


def test_mode_decomposition_dictionary_lifts_observables():
    """Test that extended DMD runs on the observables of a dictionary, which become the states of the surrogate."""
    trajectory = _linear_trajectory()
    decomposition = _fit(
        trajectory,
        variables=['linear.x', 'linear.y'],
        dictionary=PolynomialFeatures(2, include_bias=False),
    )
    assert decomposition.labels_ == ['linear.x', 'linear.y', 'linear.x^2', 'linear.x linear.y', 'linear.y^2']
    assert decomposition.states_ == ['x', 'y', 'x_2', 'x_y', 'y_2']
    assert PassCompiler().check().prepare(decomposition.system_).report_.ok


def test_mode_decomposition_fits_several_series():
    """Test that DMD fits the pairs of several runs of the same system, given as a list or a 3-D array."""
    times = np.arange(0.0, 5.0, 0.1)
    runs = [(start * np.exp(-0.5 * times))[:, None] for start in (1.0, -2.0)]
    decomposition = _fit(runs, variables=['x'], dt=0.1)
    assert _fit(np.stack(runs), variables=['x'], dt=0.1).eigenvalues_.tolist() == decomposition.eigenvalues_.tolist()
    assert decomposition.eigenvalues_.real.round(8).tolist() == [-0.5]
    assert decomposition.system_.elements[0].x == 1.0


def test_mode_decomposition_rejects_series_of_other_variables():
    """Test that series of different variables raise an error."""
    trajectory = _linear_trajectory()
    analysis = ModeDecompositionAnalysis().check()
    with pytest.raises(AnalysisError, match='same variables'):
        analysis.prepare([trajectory, np.zeros((10, 3))])


def test_mode_decomposition_rejects_no_series():
    """Test that an empty sequence of series raises an error."""
    with pytest.raises(AnalysisError, match='No series'):
        ModeDecompositionAnalysis().check().prepare([])


def test_mode_decomposition_rejects_validation_share_of_one():
    """Test that a validation share outside `[0, 1)` raises an error."""
    with pytest.raises(AnalysisError, match='validation share'):
        ModeDecompositionAnalysis(validation=1.0).check()
