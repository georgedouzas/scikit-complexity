"""Test the `_lorenz` module of `skcomplex.systems.mathematics.common.flows`."""

import numpy as np

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.common.flows import LorenzElement


def test_lorenz_element_settles_on_a_fixed_point_below_the_first_bifurcation():
    """Test that the Lorenz flow with `rho < 1` decays to the origin."""
    system = CompositeSystem(elements=[LorenzElement('lorenz', rho=0.5)])
    trajectory = simulate(system, until=30.0, dt=1.0)
    final = [trajectory.values(f'lorenz.{name}')[-1] for name in 'xyz']
    np.testing.assert_allclose(final, 0.0, atol=1e-3)


def test_lorenz_element_reaches_the_convection_fixed_point():
    """Test that the Lorenz flow with `1 < rho < 24.74` settles on `x = y = sqrt(beta (rho - 1))`, `z = rho - 1`."""
    system = CompositeSystem(elements=[LorenzElement('lorenz', rho=10.0)])
    trajectory = simulate(system, until=60.0, dt=1.0)
    beta = 8 / 3
    np.testing.assert_allclose(trajectory.values('lorenz.x')[-1], np.sqrt(beta * 9.0), rtol=1e-3)
    np.testing.assert_allclose(trajectory.values('lorenz.z')[-1], 9.0, rtol=1e-3)
