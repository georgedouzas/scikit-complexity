"""Define the Lennard-Jones pair potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, oo, sqrt
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.laws.utils import neighbors
from .....core.components.spaces import RadiusNeighborhood
from .....core.types import Role


class LennardJonesInteraction(Interaction):
    """Lennard-Jones pair potential without a shift.

    The potential is `U(r) = 4 epsilon ((sigma / r)^12 - (sigma / r)^6)`, and every member feels the force `-U'(r)`
    along the line of centres from each member within the cutoff, `x_i'' += sum_j -U'(r_ij) (x_i - x_j) / (r_ij m_i)`.
    In a periodic box the separations are minimum images. It is the force between atoms of a noble gas, a model fluid
    of molecular dynamics, or colloids with a short-range attraction.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of particles.

        cutoff:
            The distance beyond which pairs do not interact.


        epsilon:
            The depth of the potential well.

        sigma:
            The distance at which the potential is zero.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import LennardJonesInteraction
        >>> LennardJonesInteraction('lj', target='gas', cutoff=2.5).sigma
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        cutoff: Annotated[float, RealSet.open(0, oo)] = 1.0,
        epsilon: Annotated[float, RealSet.open(0, oo)] = 1.0,
        sigma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.epsilon = epsilon
        self.sigma = sigma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the force of the neighbours within the cutoff on every particle.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, x = context, context.target.x
        separation = x - s.target.other.x
        r = sqrt(separation.dot_product(separation))
        distance = SR.var('r', domain='positive')
        energy = 4 * s.epsilon * ((s.sigma / distance) ** 12 - (s.sigma / distance) ** 6)
        slope = energy.diff(distance).subs({distance: r})
        force = neighbors(-slope * separation / r, RadiusNeighborhood(self.cutoff))
        self.laws_ = [RateLaw(x.diff(s.t, 2), force / s.target.m)]
