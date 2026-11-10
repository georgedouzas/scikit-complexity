"""Test the `_gels` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.active_matter import ActiveViscoelasticGelSystem


def _force_balance_residual(density, displacement, velocity, spacing):
    """Return the largest residual of `(gamma - eta d_xx) v = d_x (E u_x + Ta(c))` on the stencils of the field."""

    def central(field):
        return (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / (2 * spacing)

    stress = 0.1 * central(displacement) + 4.0 * density / (density + 1.0)
    second = (np.roll(velocity, -1, axis=-1) - 2 * velocity + np.roll(velocity, 1, axis=-1)) / spacing**2
    return np.abs(velocity - second - central(stress)).max()


def test_active_viscoelastic_gel_concentrates_above_the_contractile_threshold():
    """Test that the cells concentrate into peaks while the velocity solves the force balance at every step."""
    trajectory = simulate(ActiveViscoelasticGelSystem(), random_state=0)
    density = np.asarray(trajectory.get_values('gel.c'))
    displacement = np.asarray(trajectory.get_values('gel.u'))
    velocity = np.asarray(trajectory.get_values('v'))
    assert density[-1].max() > 3.0
    assert density[-1].min() > 0.0
    assert abs(density[-1].sum() - density[0].sum()) < 1e-9
    assert np.abs(velocity).max() > 0.1
    assert _force_balance_residual(density, displacement, velocity, 10.0 / 128) < 1e-10


def test_active_viscoelastic_gel_stays_uniform_below_the_contractile_threshold():
    """Test that a perturbation decays when `c Ta'(c)` is below `gamma D`."""
    density = np.asarray(
        simulate(ActiveViscoelasticGelSystem(T0=0.2), random_state=0).get_values('gel.c'),
    )
    assert np.ptp(density[-1]) < np.ptp(density[0])
