"""Test the `_rigid_bodies` module of `skcomplex.systems.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ParameterError
from skcomplex.systems.physics.soft_matter import AnisotropicBrownianParticlesSystem, RigidDimersSystem


def _bodies(n_particles=400, **settings):
    """Return the times, positions, axes, and spins of a run of anisotropic particles to time 1."""
    trajectory = simulate(
        AnisotropicBrownianParticlesSystem(n_particles=n_particles, **settings),
        until=1.0,
        random_state=0,
    )
    positions = np.stack([np.asarray(trajectory.get_values(f'particles.{name}')) for name in 'xyz'], axis=-1)
    axes = np.stack([np.asarray(trajectory.get_values(f'particles.u{name}')) for name in 'xyz'], axis=-1)
    return np.asarray(trajectory.times), positions, axes, np.asarray(trajectory.get_values('particles.psi'))


def test_janus_axes_stay_on_the_sphere_and_decorrelate_as_exp_minus_two_d_r_t():
    """Test that the axes keep unit length and that `<u(t) .

    u(0)>` decays as `exp(-2 D_r t)` in three dimensions.
    """
    times, _, axes, _ = _bodies(2000)
    assert np.abs(np.linalg.norm(axes, axis=-1) - 1).max() < 1e-12
    correlation = (axes * axes[0]).sum(axis=-1).mean(axis=1)
    np.testing.assert_allclose(correlation[[5, 10, 20]], np.exp(-1.5 * times[[5, 10, 20]]), atol=0.02)


def test_ellipsoid_and_chiral_particles_follow_their_body_frame_diffusion_tensor():
    """Test the short-time diffusion along and across the axis, and the coupling of translation and spin."""
    times, positions, axes, spins = _bodies(D_par=1.5, D_perp=0.5, D_r=0.5, D_c=0.5)
    steps = np.diff(positions, axis=0)[:20]
    along = (steps * axes[:20]).sum(axis=-1)
    across = (steps**2).sum(axis=-1) - along**2
    interval = times[1]
    assert np.mean(along**2) / (2 * interval) == pytest.approx(1.5, rel=0.1)
    assert np.mean(across) / (4 * interval) == pytest.approx(0.5, rel=0.1)
    assert np.mean(along * np.diff(spins, axis=0)[:20]) / (2 * interval) == pytest.approx(0.5, rel=0.15)


def test_anisotropic_particles_refuse_a_coupling_beyond_the_positive_tensor():
    """Test that a coupling larger than `sqrt(D_par D_spin)` is refused."""
    with pytest.raises(ParameterError):
        AnisotropicBrownianParticlesSystem(D_par=1.0, D_spin=0.25, D_c=0.6).prepare()


@pytest.mark.parametrize(('v0', 'expected'), [(0.0, 0.05), (2.0, 0.05 + 4.0 / 4)])
def test_rigid_dimers_diffuse_as_their_free_draining_limit(v0, expected):
    """Test the rod's rotational diffusion `2 D_t / L²` and the long-time diffusion `D_t / 2 + v0² / (4 D_r)`."""
    trajectory = simulate(RigidDimersSystem(n_dimers=400, v0=v0), until=20.0, random_state=0)
    times = np.asarray(trajectory.times)
    centres, angles = np.asarray(trajectory.get_values('dimers.c')), np.asarray(trajectory.get_values('dimers.phi'))
    diffusion = ((centres[-1] - centres[100]) ** 2).sum(axis=-1).mean() / (4 * (times[-1] - times[100]))
    assert diffusion == pytest.approx(expected, rel=0.2)
    if v0 == 0:
        assert ((angles[10] - angles[0]) ** 2).mean() / (2 * times[10]) == pytest.approx(2 * 0.1 / 1.0, rel=0.15)
    assert math.isfinite(diffusion)
