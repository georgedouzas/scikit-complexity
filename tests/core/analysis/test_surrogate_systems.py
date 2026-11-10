"""Test the state names and the sampling shared by the surrogates fitted to series."""

import numpy as np
import pytest

from skcomplex.core.analysis import mode_decomposition, reservoir, sparse_identification
from skcomplex.core.engine.runtime import simulate


def _fit(labels, dt=0.1):
    """Return the mode decomposition of decaying series of the given variables, sampled every `dt`."""
    times = np.arange(30) * dt
    values = np.exp(-np.outer(times, np.arange(1, len(labels) + 1)))
    return mode_decomposition(values, variables=labels, dt=dt)


@pytest.mark.parametrize(
    ('labels', 'states'),
    [
        (['a.x', 'b.y'], ['x', 'y']),
        (['a.x', 'b.x'], ['a_x', 'b_x']),
        (['cells.n[0]', 'cells.n[1]'], ['n_0', 'n_1']),
        (['lambda', 'label'], ['lambda_', 'label_']),
        (['variables'], ['variables_']),
        (['0', '1'], ['v_0', 'v_1']),
    ],
)
def test_surrogate_names_states_after_variables(labels, states):
    """Test that the states take the last part of every label, or the whole label where those clash."""
    assert _fit(labels).states == states


def test_surrogate_accepts_steps_with_rounding_noise():
    """Test that steps equal up to rounding count as uniform."""
    assert _fit(['x']).dt == pytest.approx(0.1)


def _damped_rotation():
    """Return a damped rotation of two variables, sampled every 0.05."""
    times = 0.05 * np.arange(200.0)
    return np.column_stack([np.exp(-0.5 * times) * np.cos(2 * times), np.exp(-0.5 * times) * np.sin(2 * times)])


def _last_values(analysis, states, until, dt):
    """Return the last values of states of the surrogate of an analysis."""
    trajectory = simulate(analysis.system, until=until, dt=dt)
    return [float(np.ravel(trajectory.get_values(f'surrogate.{state}'))[-1]) for state in states]


def test_sparse_identification_keeps_its_reference_values():
    """Test the run of the sparse surrogate against the values recorded before it moved to declared states."""
    analysis = sparse_identification(_damped_rotation(), variables=['x', 'y'], degree=1, threshold=0.01, dt=0.05)
    assert _last_values(analysis, ['x', 'y'], 1.0, 0.01) == pytest.approx(
        [-0.25163852889586635, 0.5533559045838776],
        rel=1e-9,
    )


def test_mode_decomposition_keeps_its_reference_values():
    """Test the run of the linear surrogate against the values recorded before it moved to declared states."""
    analysis = mode_decomposition(_damped_rotation(), variables=['x', 'y'], dt=0.05)
    assert _last_values(analysis, ['x', 'y'], 1.0, 0.01) == pytest.approx(
        [-0.25240559441454513, 0.5515162392689774],
        rel=1e-9,
    )


def test_reservoir_keeps_its_reference_values():
    """Test the run of the reservoir surrogate against the values recorded before it moved to declared states."""
    analysis = reservoir(
        _damped_rotation(),
        variables=['x', 'y'],
        kind='next_generation',
        delay=2,
        order=2,
        ridge=1e-6,
        validation=0.0,
        dt=0.05,
    )
    assert _last_values(analysis, ['x', 'y'], 10, 1) == pytest.approx(
        [-0.0024828568048470227, 0.004773226924693426],
        rel=1e-9,
    )
