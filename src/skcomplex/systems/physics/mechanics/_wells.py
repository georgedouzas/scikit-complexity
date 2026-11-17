"""Potential well systems.

The module holds a particle in the one-dimensional well `k x² e^-x`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class ExponentialWellInteraction(Interaction):
    """Potential `k x² e^-x` on a particle in one dimension.

    The potential has a well at `x = 0` behind a barrier at `x = 2`. The particle of mass `m` follows
    `m x'' = -k (2 x - x²) e^-x`.

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

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the particle, the force `-dU/dx` over its mass.

        Args:
            context:
                The namespace of the particle, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            force = -context.k * (2 * x - x**2) * exp(-x)
            self.laws_ = [RateLaw(x.diff(context.t, 2), force / context.target.m)]


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

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'potential', 'hamiltonian')
    references = ('Starkovich 2021',)
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        v: float = 0.5,
        *,
        k: float = 2.0,
        m: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.k = k
        self.m = m

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle moving without friction in the potential `U(x) = k x² e^-x`.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('particle', values={'x': self.x, 'm': self.m})],
            interactions=[ExponentialWellInteraction('well', target='particle', k=self.k)],
        )
        self.initial_state_ = {"particle.x'": self.v}
