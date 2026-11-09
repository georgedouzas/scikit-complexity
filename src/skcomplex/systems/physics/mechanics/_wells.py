"""Potential well systems.

The module holds a particle in the one-dimensional well `k x² e^-x`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec
from ..common.particles import MassPointElement, PotentialEnergyInteraction


class ExponentialWellInteraction(PotentialEnergyInteraction):
    """Potential `k x² e^-x` on a particle in one dimension.

    The potential has a well at `x = 0` behind a barrier at `x = 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The strength of the potential.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ExponentialWellInteraction
        >>> ExponentialWellInteraction('well', target='particle', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, target=target)
        self.k = k

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `k x² e^-x`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        position = s.target.x[0]
        return s.k * position**2 * exp(-position)


class ParticleInAWellSystem(BaseSystem):
    """A particle moving without friction in the potential `U(x) = k x² e^-x`.

    The potential has a stable equilibrium at `x = 0` and an unstable one at `x = 2`, the top of a barrier of height `4
    k e^-2`. A particle in the well with less energy than the barrier oscillates, and one with more escapes to large
    `x`. The system has the particle `particle` and its position `particle.x` in one dimension.

    Args:
        x:
            The initial position.

        v:
            The initial velocity.

        k:
            The strength of the potential.

        m:
            The mass of the particle.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParticleInAWellSystem
        >>> ParticleInAWellSystem(k=3.0).prepare().system_.interactions[0].k
        3.0
    """

    field = "physics/mechanics"
    topics = ("mechanics", "ode", "potential", "hamiltonian")
    references = ("Starkovich 2021",)
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        v: float = 0.5,
        *,
        k: float = 2.0,
        m: float = 1.0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.k = k
        self.m = m

    def _build_system(self) -> System:
        """Return a particle moving without friction in the potential `U(x) = k x² e^-x`."""
        return System(
            self.label,
            elements=[MassPointElement("particle", x=self.x, x_d1=self.v, m=self.m)],
            interactions=[ExponentialWellInteraction("well", target="particle", k=self.k)],
            space=ContinuousSpace(n_dim=1),
        )
