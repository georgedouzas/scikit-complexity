"""Test the kinetic theory of flocking of Bertin, Droz, and Grégoire."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.active_matter import BertinDrozGregoireSystem


def polarization_change(density: float, transition: float) -> float:
    """Return the ratio of the final and initial polarization moments of a homogeneous flock."""
    angles = 32
    system = BertinDrozGregoireSystem(1, angles, density=density * transition, interaction_range=0.5, noise=0.5)
    trajectory = simulate(system, until=2.0, dt=2.0)
    values = np.asarray(trajectory.values('flock.f'))
    theta = 2 * np.pi * (np.arange(angles) + 0.5) / angles
    return float((values[-1] @ np.cos(theta)) / (values[0] @ np.cos(theta)))


@pytest.mark.parametrize(('factor', 'grows'), [(0.8, False), (1.2, True)])
def test_flocking_onset_at_the_transition_density(factor: float, grows: bool) -> None:
    """Decay the polarization 20 percent below the transition density and grow it 20 percent above."""
    noise = 0.5
    survival = np.exp(-(noise**2) / 2)
    transition = np.pi * (1 - survival) / (8 * 0.5 * (survival - 2 / 3))
    assert (polarization_change(factor, transition) > 1.0) is grows
