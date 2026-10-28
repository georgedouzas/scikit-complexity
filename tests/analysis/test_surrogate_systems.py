"""Test the state names and the sampling shared by the surrogates fitted to series."""

import numpy as np
import pytest

from skcomplex.analysis import ModeDecompositionAnalysis


def _fit(labels, dt=0.1):
    """Return the mode decomposition of decaying series of the given variables, sampled every `dt`."""
    times = np.arange(30) * dt
    values = np.exp(-np.outer(times, np.arange(1, len(labels) + 1)))
    return ModeDecompositionAnalysis(labels, dt=dt).check().prepare(values).run()


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
    assert _fit(labels).states_ == states


def test_surrogate_accepts_steps_with_rounding_noise():
    """Test that steps equal up to rounding count as uniform."""
    assert _fit(['x']).dt_ == pytest.approx(0.1)
