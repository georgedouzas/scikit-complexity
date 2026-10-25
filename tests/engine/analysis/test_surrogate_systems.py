"""Test the state names and the sampling shared by the surrogates fitted to series."""

import numpy as np
import pytest

from skcomplex.engine.analysis import DMDSurrogate, Series


def _series(labels):
    """Return decaying series of the given variables, sampled every 0.1."""
    times = np.arange(0.0, 3.0, 0.1)
    rates = np.arange(1, len(labels) + 1)
    return Series(times, np.exp(-np.outer(times, rates)), labels)


@pytest.mark.parametrize(
    ('labels', 'states'),
    [
        (['a.x', 'b.y'], ['x', 'y']),
        (['a.x', 'b.x'], ['a_x', 'b_x']),
        (['cells.n[0]', 'cells.n[1]'], ['n_0', 'n_1']),
        (['lambda', 'label'], ['lambda_', 'label_']),
        (['0', '1'], ['v_0', 'v_1']),
    ],
)
def test_surrogate_names_states_after_variables(labels, states):
    """Test that the states take the last part of every label, or the whole label where those clash."""
    assert DMDSurrogate().fit(_series(labels)).states_ == states


def test_surrogate_accepts_steps_with_rounding_noise():
    """Test that steps equal up to rounding count as uniform."""
    times = np.arange(30) * 0.1
    decomposition = DMDSurrogate().fit(Series(times, np.exp(-times)[:, None], ['x']))
    assert decomposition.dt_ == pytest.approx(0.1)
