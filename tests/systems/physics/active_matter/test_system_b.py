"""Test the `_system_b` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np

from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.physics.active_matter import ActiveScalarElement, active_system_b


def _bubble_areas(zeta):
    """Return the areas of a small and a large bubble of the dilute phase, after relaxation and at the end."""
    rows, columns = np.indices((24, 48))
    bubbles = (np.hypot(rows - 12, columns - 12) < 4) | (np.hypot(rows - 12, columns - 36) < 7)
    values = {'phi': np.where(bubbles, -1.0, 1.0), 'zeta': zeta, 'D': 0.0}
    system = System(
        elements=[FieldElement('fluid', element=ActiveScalarElement, space=LatticeSpace((24, 48)), values=values)],
    )
    trajectory = Simulation(system, dt=50.0, solver=EulerMaruyamaSolver(dt=0.02)).prepare().run(until=300).trajectory_
    phi = trajectory.lattice('fluid.phi')[[1, -1]] < 0
    return phi[:, :, :24].sum(axis=(1, 2)), phi[:, :, 24:].sum(axis=(1, 2))


def test_active_system_b_keeps_the_mean_and_phase_separates():
    """Test that Active System B+ keeps the mean of phi under its conserved noise and separates into two phases."""
    system = active_system_b((32, 32))
    trajectory = (
        Simulation(system, dt=50.0, solver=EulerMaruyamaSolver(dt=0.02), random_state=0)
        .prepare()
        .run(until=300)
        .trajectory_
    )
    phi = trajectory.lattice('fluid.phi')
    np.testing.assert_allclose(phi.mean(axis=(1, 2)), phi[0].mean(), rtol=1e-9)
    assert phi[-1].max() > 0.9
    assert phi[-1].min() < -0.9


def test_active_system_b_reverses_ostwald_ripening():
    """Test that the small bubble dissolves into the large one without zeta and grows at its expense with zeta = 4."""
    (small_start, small_end), (large_start, large_end) = _bubble_areas(0.0)
    assert small_end == 0
    assert large_end > large_start
    (small_start, small_end), (large_start, large_end) = _bubble_areas(4.0)
    assert small_end > small_start
    assert large_end < large_start
