"""Test the `_heat` module of `skcomplex.systems.physics.heat`."""

import numpy as np
import pytest

from skcomplex.systems.physics.heat import HeatDiffusionSystem, HeatLShapeSystem, harmonic


@pytest.mark.parametrize(('boundary', 'kept'), [('zero-flux', True), ('periodic', True), ('fixed', False)])
def test_heat_diffusion_keeps_its_heat_unless_the_edges_are_held_cold(boundary, kept):
    """Test that insulated and periodic plates keep their total heat and a plate with cold edges loses it."""
    trajectory = HeatDiffusionSystem((16, 16), boundary=boundary).prepare().simulate(until=20.0, dt=10.0)
    totals = trajectory.get_values('plate.u').sum(axis=1)
    assert bool(totals[-1] == pytest.approx(totals[0])) is kept


def test_heat_on_the_l_shape_relaxes_to_the_harmonic_temperature():
    """Test that the temperature of the L-shaped plate approaches the harmonic temperature of its edges."""
    system = HeatLShapeSystem(2).prepare()
    trajectory = system.simulate(until=4.0, dt=4.0)
    nodes = system.system_.elements[0].space.prepare().positions_
    exact = harmonic(nodes[:, 0], nodes[:, 1])
    assert np.abs(trajectory.get_values('plate.u')[-1] - exact).max() < 0.05
