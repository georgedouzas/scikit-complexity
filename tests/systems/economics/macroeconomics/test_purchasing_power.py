"""Test the `_purchasing_power` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.systems.economics.macroeconomics import CategorizedPurchasingPowerSystem
from skcomplex.systems.economics.macroeconomics._purchasing_power import A_MATRIX, B_MATRIX, C_MATRIX, D_MATRIX

from ..conftest import _run


def _indices(feedback):
    """Return the indices of the sectors after a step policy on the luxury sector."""
    trajectory = _run(CategorizedPurchasingPowerSystem(feedback=feedback), 30.0, 0.05)
    return [trajectory.values(f'economy.y{i}') for i in (1, 2, 3)]


def test_categorized_purchasing_power_feedback_keeps_policies_in_their_sector():
    """Test that with feedback only the luxury index moves, to the steady state of its decoupled system."""
    basic, desired, luxury = _indices(feedback=True)
    assert np.abs(basic).max() < 1e-8
    assert np.abs(desired).max() < 1e-8
    A, B, C, D = (np.asarray(matrix) for matrix in (A_MATRIX, B_MATRIX, C_MATRIX, D_MATRIX))
    assert luxury[-1] == pytest.approx(C[2] @ np.linalg.solve(-A, B[:, 2]) + D[2, 2], rel=2e-3)


def test_categorized_purchasing_power_without_feedback_spills_over():
    """Test that without feedback a policy of the luxury sector moves the basic and desired indices too."""
    basic, desired, _ = _indices(feedback=False)
    assert np.abs(basic).max() > 0.1
    assert np.abs(desired).max() > 0.1
