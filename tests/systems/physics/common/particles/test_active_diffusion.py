"""Test the `_active_diffusion` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import ActiveDiffusionInteraction, ActiveParticleElement


def test_position_and_orientation_diffuse_and_the_orientation_turns():
    """Test that the position spreads as `4 D_t t` and the orientation drifts as `omega t` and spreads as `2 D_r t`."""
    swimmers = ActiveParticleElement('swimmers', x=np.zeros((4000, 2)), size=4000, space=ContinuousSpace(n_dim=2))
    diffusion = ActiveDiffusionInteraction('diffusion', target='swimmers', D_t=0.5, D_r=0.25, omega=1.0)
    system = System(elements=[swimmers], interactions=[diffusion], solver=EulerMaruyamaSolver(dt=0.1)).prepare()
    trajectory = system.simulate(until=2.0, dt=2.0, random_state=0)
    positions, angles = trajectory.get_values('swimmers.x')[-1], trajectory.get_values('swimmers.theta')[-1]
    assert (positions**2).sum(axis=1).mean() == pytest.approx(4 * 0.5 * 2.0, rel=0.05)
    assert angles.mean() == pytest.approx(2.0, abs=0.05)
    assert angles.var() == pytest.approx(2 * 0.25 * 2.0, rel=0.05)
