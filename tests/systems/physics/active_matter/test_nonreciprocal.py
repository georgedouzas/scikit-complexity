"""Test the `_nonreciprocal` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.active_matter import NonreciprocalVicsekSystem


def _species_headings(coupling):
    """Return the mean headings of species A and B of a nonreciprocal Vicsek run, one complex number per frame."""
    trajectory = simulate(
        NonreciprocalVicsekSystem(J_ab=coupling[0], J_ba=coupling[1]),
        backend=NumbaBackend(),
        random_state=0,
    )
    return trajectory.get_values('heading_a'), trajectory.get_values('heading_b')


def test_nonreciprocal_vicsek_chases_and_rotates():
    """Test that reciprocal alignment flocks both species together and antialignment locks them a quarter turn apart.

    Above the chiral threshold the mean headings of A and B stay a quarter turn apart and rotate together, the chiral
    phase of Fruchart et al. (2021), while reciprocal alignment orders both species in one direction that stays put.
    """
    heading_a, heading_b = _species_headings((1.0, 1.0))
    assert np.abs(heading_a[-100:]).mean() > 0.9
    assert np.abs(np.angle(heading_a[-100:] * np.conj(heading_b[-100:]))).mean() < 0.2
    assert abs(np.unwrap(np.angle(heading_a))[-1] - np.unwrap(np.angle(heading_a))[-150]) < 0.5
    heading_a, heading_b = _species_headings((1.0, -1.0))
    relative = np.angle(heading_a[-100:] * np.conj(heading_b[-100:]))
    assert np.abs(np.abs(relative).mean() - np.pi / 2) < 0.3
    assert abs(np.unwrap(np.angle(heading_a))[-1] - np.unwrap(np.angle(heading_a))[-150]) > 20
