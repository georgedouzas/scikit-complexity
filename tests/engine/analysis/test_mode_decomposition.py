"""Test linear surrogates fitted by dynamic mode decomposition and extended DMD."""

import math

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.core.base import AnalysisError
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import RateLaw
from skcomplex.core.space import ContinuousTime
from skcomplex.core.system import System
from skcomplex.engine.analysis import BaseSurrogate, DMDSurrogate, Series
from skcomplex.engine.compile import compile_system, validate
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver, resolve_solver

MATRIX = np.asarray([[-0.1, 2.0, 0.0], [-2.0, -0.3, 0.5], [0.0, 0.0, -0.5]])

LinearComponent = ElementComponent.define(
    'LinearComponent',
    states={'x': 1.0, 'y': 0.5, 'z': -1.0},
    laws=lambda s: [
        RateLaw(s.x, float(MATRIX[0, 0]) * s.x + float(MATRIX[0, 1]) * s.y),
        RateLaw(s.y, float(MATRIX[1, 0]) * s.x + float(MATRIX[1, 1]) * s.y + float(MATRIX[1, 2]) * s.z),
        RateLaw(s.z, float(MATRIX[2, 2]) * s.z),
    ],
)


def _linear_trajectory():
    """Return a trajectory of the linear system `dx/dt = A x`, sampled every 0.05 for 10 time units."""
    solver = ScipyODESolver(method='DOP853', rtol=1e-12, atol=1e-12)
    return (
        Simulation(System(components=[LinearComponent('linear')]), solver=solver, dt=0.05).run(until=10.0).trajectory_
    )


def _linear_series(trajectory, samples):
    """Return the series of the first samples of the states of the linear trajectory."""
    names = ['linear.x', 'linear.y', 'linear.z']
    values = np.column_stack([trajectory.values(name)[:samples] for name in names])
    return Series(trajectory.times[:samples], values, names)


def test_dmd_surrogate_recovers_linear_eigenvalues():
    """Test that the continuous eigenvalues of DMD equal those of the system matrix within 1e-6."""
    decomposition = DMDSurrogate().fit(_linear_trajectory())
    expected = np.sort_complex(np.linalg.eigvals(MATRIX))
    assert np.abs(np.sort_complex(decomposition.eigenvalues_) - expected).max() < 1e-6
    assert np.allclose(decomposition.generator_, MATRIX, atol=1e-6)


def test_dmd_surrogate_surrogate_validates_under_continuous_time():
    """Test that the surrogate passes validation, declares continuous time, and runs with an ODE solver."""
    trajectory = _linear_trajectory()
    decomposition = DMDSurrogate().fit(trajectory)
    assert validate(decomposition.system_).ok
    assert isinstance(decomposition.system_.time, ContinuousTime)
    assert isinstance(resolve_solver(compile_system(decomposition.system_), 'auto'), ScipyODESolver)
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
    run = Simulation(decomposition.system_, solver=solver, dt=0.05).run(until=10.0).trajectory_
    assert np.allclose(run.values('surrogate.y'), trajectory.values('linear.y'), atol=1e-6)


def test_dmd_surrogate_run_continues_trajectory():
    """Test that running the modes continues the trajectory after the last sample it was fitted to."""
    trajectory = _linear_trajectory()
    decomposition = DMDSurrogate().fit(_linear_series(trajectory, 101))
    assert decomposition.labels_ == ['linear.x', 'linear.y', 'linear.z']
    assert decomposition.states_ == ['x', 'y', 'z']
    forecast = decomposition.run(50)
    assert forecast.shape == (50, 3)
    assert np.allclose(forecast[:, 2], trajectory.values('linear.z')[101:151], atol=1e-8)


def test_dmd_surrogate_run_matches_simulated_system():
    """Test that the forecast of the modes equals the simulation of the fitted system that the base runs."""
    decomposition = DMDSurrogate().fit(_linear_series(_linear_trajectory(), 101))
    simulated = BaseSurrogate._run(decomposition, 20)
    assert np.allclose(decomposition.run(20), simulated, atol=1e-4)


def test_dmd_surrogate_builds_without_checking():
    """Test that an invalid validation share builds and fails when checked."""
    surrogate = DMDSurrogate(validation=1.0)
    assert surrogate.get_params()['validation'] == 1.0
    with pytest.raises(AnalysisError, match='validation share'):
        surrogate.check()


def test_dmd_surrogate_reports_small_errors_on_linear_data():
    """Test that the training and validation errors of a linear system vanish."""
    decomposition = DMDSurrogate().fit(_linear_trajectory())
    assert decomposition.training_error_ < 1e-8
    assert decomposition.validation_error_ < 1e-8


def test_dmd_surrogate_without_validation_has_no_validation_error():
    """Test that fitting every sample leaves the validation error undefined."""
    assert math.isnan(DMDSurrogate(validation=0.0).fit(_linear_trajectory()).validation_error_)


def test_dmd_surrogate_rank_truncates_modes():
    """Test that a rank keeps that many modes."""
    decomposition = DMDSurrogate(rank=2).fit(_linear_trajectory())
    assert decomposition.modes_.shape == (3, 2)
    assert decomposition.eigenvalues_.shape == (2,)


def test_dmd_surrogate_dictionary_lifts_observables():
    """Test that extended DMD runs on the observables of a dictionary, which become the states of the surrogate."""
    trajectory = _linear_trajectory()
    decomposition = DMDSurrogate(['linear.x', 'linear.y'], dictionary=PolynomialFeatures(2, include_bias=False)).fit(
        trajectory,
    )
    assert decomposition.labels_ == ['linear.x', 'linear.y', 'linear.x^2', 'linear.x linear.y', 'linear.y^2']
    assert decomposition.states_ == ['x', 'y', 'x_2', 'x_y', 'y_2']
    assert validate(decomposition.system_).ok


def test_dmd_surrogate_fits_several_series():
    """Test that DMD fits the pairs of several series of the same system."""
    times = np.arange(0.0, 5.0, 0.1)
    series = [Series(times, (start * np.exp(-0.5 * times))[:, None], ['x']) for start in (1.0, -2.0)]
    decomposition = DMDSurrogate().fit(series)
    assert decomposition.eigenvalues_.real.round(8).tolist() == [-0.5]
    assert decomposition.system_.elements[0].x == 1.0


def test_dmd_surrogate_rejects_uneven_sampling():
    """Test that series sampled at uneven times raise an error."""
    times = np.asarray([0.0, 0.1, 0.3, 0.4])
    with pytest.raises(AnalysisError, match='uniform step'):
        DMDSurrogate().fit(Series(times, times[:, None], ['x']))


def test_dmd_surrogate_rejects_series_of_other_variables():
    """Test that series of different variables raise an error."""
    times = np.arange(0.0, 1.0, 0.1)
    with pytest.raises(AnalysisError, match='same variables'):
        DMDSurrogate().fit([Series(times, times[:, None], ['x']), Series(times, times[:, None], ['y'])])


def test_dmd_surrogate_rejects_no_series():
    """Test that an empty sequence of series raises an error."""
    with pytest.raises(AnalysisError, match='No series'):
        DMDSurrogate().fit([])


def test_dmd_surrogate_rejects_validation_share_of_one():
    """Test that a validation share outside `[0, 1)` raises an error."""
    times = np.arange(0.0, 1.0, 0.1)
    with pytest.raises(AnalysisError, match='validation share'):
        DMDSurrogate(validation=1.0).fit(Series(times, times[:, None], ['x']))
