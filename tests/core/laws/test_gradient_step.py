"""Test the `_gradient_step` module of `skcomplex.core.laws`."""

import numpy as np

from skcomplex.core.laws import GradientStepLaw
from skcomplex.specs import DatasetSpec, ObservableSpec


def _law():
    """Return a gradient-step law of the readout weights."""
    data = DatasetSpec({'inputs.x': np.zeros((4, 3, 1))})
    return GradientStepLaw(['readout.W'], ObservableSpec('error', 0.0), data, 'adam', 2)


def test_gradient_step_law_changes_no_state():
    """Test that the law changes no state of the system that holds it."""
    assert _law().prepare('training').targets_ == ()
