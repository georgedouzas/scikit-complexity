"""Test the `_potential` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np

from skcomplex.core.elements import ParticleElement
from skcomplex.core.interactions import PotentialInteraction
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.specs import ParameterSpec, RoleSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver


class SpringInteraction(PotentialInteraction):
    """Spring of a stiffness between two particles."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[str, RoleSpec()],
        b: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.k = k

    def _get_energy(self, s):
        """Return the energy of the stretched spring."""
        gap = s.a.x - s.b.x
        return s.k * gap.dot_product(gap) / 2


def test_spring_oscillates_its_particles_about_their_centre_of_mass():
    """Test that two particles joined by a spring keep their centre of mass and oscillate at their frequency."""
    system = System(
        elements=[ParticleElement('a', x=[-1.0], m=1.0), ParticleElement('b', x=[1.0], m=1.0)],
        interactions=[SpringInteraction('spring', a='a', b='b', k=0.5)],
        space=ContinuousSpace(n_dim=1),
    )
    trajectory = Simulation(system, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=0.1).check().run(until=np.pi)
    a, b = trajectory.trajectory_.values('a.x')[:, 0], trajectory.trajectory_.values('b.x')[:, 0]
    np.testing.assert_allclose(a + b, 0.0, atol=1e-9)
    np.testing.assert_allclose(b - a, 2.0 * np.cos(trajectory.trajectory_.times), atol=1e-6)
