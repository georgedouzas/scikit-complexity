"""Test the `_springs` module of `skcomplex.systems.physics.mechanics`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.mechanics import ElasticLinkInteraction, TwoMassSpringDamperSystem

from ..conftest import _run, _series


def test_two_mass_spring_damper_conserves_energy_without_damping():
    """Test that the two masses between walls keep their energy without dampers and forces."""
    trajectory = _run(TwoMassSpringDamperSystem(k=2.0), 20.0, 0.1)
    x1, x2, v1, v2 = _series(trajectory, 'first.x', 'second.x', "first.x'", "second.x'")
    energy = 0.5 * (v1**2 + v2**2) + 0.5 * (x1 - 1) ** 2 + 0.5 * (x2 - 2) ** 2 + (x2 - x1 - 1) ** 2
    assert np.ptp(energy) < 1e-8


def test_elastic_link_conserves_energy_and_momentum():
    """Test that two masses joined by a spring keep their total energy and momentum."""
    masses = [Element('a', values={'x': 0.0, 'm': 1.0}), Element('b', values={'x': 2.0, 'm': 2.0})]
    link = ElasticLinkInteraction('spring', first='a', second='b', k=3.0, length=1.0)
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    system = System(elements=masses, interactions=[link], solver=solver).prepare()
    trajectory = system.simulate(until=5.0, dt=0.1, initial_state={"a.x'": 1.0})
    xa, xb, va, vb = (trajectory.get_values(name) for name in ('a.x', 'b.x', "a.x'", "b.x'"))
    energy = 0.5 * va**2 + 0.5 * 2.0 * vb**2 + 0.5 * 3.0 * (np.abs(xb - xa) - 1.0) ** 2
    np.testing.assert_allclose(energy, energy[0], rtol=1e-6)
    np.testing.assert_allclose(va + 2.0 * vb, 1.0, atol=1e-8)
