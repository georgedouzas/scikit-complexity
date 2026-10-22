"""Test linear surrogates fitted by dynamic mode decomposition and extended DMD."""

import math

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.core.analysis import Series, fit_dmd
from skcomplex.core.base import AnalysisError
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE, resolve_solver
from skcomplex.core.space import ContinuousTime
from skcomplex.core.system import System

MATRIX = np.asarray([[-0.1, 2.0, 0.0], [-2.0, -0.3, 0.5], [0.0, 0.0, -0.5]])

Linear = Element.define(
    'Linear',
    states={'x': 1.0, 'y': 0.5, 'z': -1.0},
    laws=lambda s: [
        Rate(s.x, float(MATRIX[0, 0]) * s.x + float(MATRIX[0, 1]) * s.y),
        Rate(s.y, float(MATRIX[1, 0]) * s.x + float(MATRIX[1, 1]) * s.y + float(MATRIX[1, 2]) * s.z),
        Rate(s.z, float(MATRIX[2, 2]) * s.z),
    ],
)


def _linear_trajectory():
    """Return a trajectory of the linear system `dx/dt = A x`, sampled every 0.05 for 10 time units."""
    solver = ScipyODE(method='DOP853', rtol=1e-12, atol=1e-12)
    return Simulation(System(components=[Linear('linear')]), solver=solver, dt=0.05).run(until=10.0).trajectory_


def test_fit_dmd_recovers_linear_eigenvalues():
    """Test that the continuous eigenvalues of DMD equal those of the system matrix within 1e-6."""
    decomposition = fit_dmd(_linear_trajectory())
    expected = np.sort_complex(np.linalg.eigvals(MATRIX))
    assert np.abs(np.sort_complex(decomposition.eigenvalues) - expected).max() < 1e-6
    assert np.allclose(decomposition.generator, MATRIX, atol=1e-6)


def test_fit_dmd_surrogate_validates_under_continuous_time():
    """Test that the surrogate passes validation, declares continuous time, and runs with an ODE solver."""
    trajectory = _linear_trajectory()
    decomposition = fit_dmd(trajectory)
    assert validate(decomposition.system).ok
    assert isinstance(decomposition.system.time, ContinuousTime)
    assert isinstance(resolve_solver(compile_system(decomposition.system), 'auto'), ScipyODE)
    solver = ScipyODE(method='DOP853', rtol=1e-10, atol=1e-10)
    run = Simulation(decomposition.system, solver=solver, dt=0.05).run(until=10.0).trajectory_
    assert np.allclose(run['surrogate.y'], trajectory['linear.y'], atol=1e-6)


def test_fit_dmd_forecast_matches_trajectory():
    """Test that the forecast of the modes reproduces the trajectory from its first sample."""
    trajectory = _linear_trajectory()
    decomposition = fit_dmd(trajectory)
    assert decomposition.labels == ['linear.x', 'linear.y', 'linear.z']
    assert decomposition.states == ['x', 'y', 'z']
    assert np.allclose(decomposition.forecast(trajectory.times)[:, 2], trajectory['linear.z'], atol=1e-8)


def test_fit_dmd_reports_small_errors_on_linear_data():
    """Test that the training and validation errors of a linear system vanish."""
    decomposition = fit_dmd(_linear_trajectory())
    assert decomposition.training_error < 1e-8
    assert decomposition.validation_error < 1e-8


def test_fit_dmd_without_validation_has_no_validation_error():
    """Test that fitting every sample leaves the validation error undefined."""
    assert math.isnan(fit_dmd(_linear_trajectory(), validation=0.0).validation_error)


def test_fit_dmd_rank_truncates_modes():
    """Test that a rank keeps that many modes."""
    decomposition = fit_dmd(_linear_trajectory(), rank=2)
    assert decomposition.modes.shape == (3, 2)
    assert decomposition.eigenvalues.shape == (2,)


def test_fit_dmd_dictionary_lifts_observables():
    """Test that extended DMD runs on the observables of a dictionary, which become the states of the surrogate."""
    trajectory = _linear_trajectory()
    decomposition = fit_dmd(trajectory, ['linear.x', 'linear.y'], dictionary=PolynomialFeatures(2, include_bias=False))
    assert decomposition.labels == ['linear.x', 'linear.y', 'linear.x^2', 'linear.x linear.y', 'linear.y^2']
    assert decomposition.states == ['x', 'y', 'x_2', 'x_y', 'y_2']
    assert validate(decomposition.system).ok


def test_fit_dmd_fits_several_series():
    """Test that DMD fits the pairs of several series of the same system."""
    times = np.arange(0.0, 5.0, 0.1)
    series = [Series(times, (start * np.exp(-0.5 * times))[:, None], ['x']) for start in (1.0, -2.0)]
    decomposition = fit_dmd(series)
    assert decomposition.eigenvalues.real.round(8).tolist() == [-0.5]
    assert decomposition.system.elements[0].x == 1.0


def test_fit_dmd_rejects_uneven_sampling():
    """Test that series sampled at uneven times raise an error."""
    times = np.asarray([0.0, 0.1, 0.3, 0.4])
    with pytest.raises(AnalysisError, match='uniform step'):
        fit_dmd(Series(times, times[:, None], ['x']))


def test_fit_dmd_rejects_series_of_other_variables():
    """Test that series of different variables raise an error."""
    times = np.arange(0.0, 1.0, 0.1)
    with pytest.raises(AnalysisError, match='same variables'):
        fit_dmd([Series(times, times[:, None], ['x']), Series(times, times[:, None], ['y'])])


def test_fit_dmd_rejects_no_series():
    """Test that an empty sequence of series raises an error."""
    with pytest.raises(AnalysisError, match='No series'):
        fit_dmd([])


def test_fit_dmd_rejects_validation_share_of_one():
    """Test that a validation share outside `[0, 1)` raises an error."""
    times = np.arange(0.0, 1.0, 0.1)
    with pytest.raises(AnalysisError, match='validation share'):
        fit_dmd(Series(times, times[:, None], ['x']), validation=1.0)
