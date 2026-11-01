"""Test the neural field models."""

import numpy as np
from scipy.optimize import brentq

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.neuroscience import amari_neural_field


def _width(potential, spacing):
    """Return the width of the region of positive potential, between its interpolated zero crossings."""
    above = np.flatnonzero(potential > 0)
    left, right = above[0], above[-1]
    start = left - potential[left] / (potential[left] - potential[left - 1])
    end = right + potential[right] / (potential[right] - potential[right + 1])
    return (end - start) * spacing


def test_amari_neural_field_holds_a_bump_of_the_predicted_width():
    """Test that a localized excitation relaxes to a stationary bump whose width solves `W(a) + h = 0`."""
    trajectory = Simulation(amari_neural_field(), dt=0.5).check().prepare().run(until=40.0).trajectory_
    potential = np.reshape(trajectory.values('field.u'), (-1, 800))
    width = _width(potential[-1], 20.0 / 800)
    root = brentq(lambda a: a * np.exp(-a) - 0.2, 1.0, 10.0)
    assert (1 - root) * np.exp(-root) < 0
    assert abs(width - root) < 0.05 * root
    assert np.abs(potential[-1] - potential[-5]).max() < 1e-6
