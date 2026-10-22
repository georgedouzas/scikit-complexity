"""Model point particles, potentials of their positions, and pair potentials of their distances."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from typing import ClassVar

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, sqrt

from ..base import (
    Domain,
    Expression,
    Requires,
    SpaceKind,
    VariableKind,
    VisualSpec,
    declarations_of,
    parameter,
    role,
    state,
    symbol,
    vector,
)
from ._components import Element, Interaction
from ._laws import Law, Rate
from ._symbols import Symbols


class Particle(Element):
    """Point particle with a position, a velocity, and a mass.

    The particle moves with its velocity, and every force on it changes its velocity by the force divided by its mass.

    Args:
        label:
            The label of the particle.

        x:
            The initial position. A number applies to every component.

        v:
            The initial velocity. A number applies to every component.

        m:
            The mass.

    Examples:
        >>> from skcomplex.core.model import Particle
        >>> Particle('ball', x=[0, 1], m=2.0)
        Particle(label='ball', m=2.0, x=[0, 1])
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.CONTINUOUS})

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x'})
    x: npt.ArrayLike = state(default=0.0, vector=True, position=True)
    v: npt.ArrayLike = state(default=0.0, vector=True, velocity_of='x')
    m: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the kinematic law `dx/dt = v`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.x, s.v)]


class Potential(Interaction):
    """Base class of interactions defined by a potential energy.

    Subclasses declare one role per particle and implement `energy`. The force on each particle is minus the
    gradient of the energy with respect to its position, derived symbolically.

    Args:
        label:
            The label of the interaction.
    """

    @abstractmethod
    def energy(self, s: Symbols) -> Expression:
        """Return the potential energy.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """

    def laws(self, s: Symbols) -> list[Law]:
        """Return the velocity change of every particle from the gradient of the energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        energy = self.energy(s)
        laws: list[Law] = []
        for name in declarations_of(self, VariableKind.ROLE):
            particle = getattr(s, name)
            gradient = vector(SR(energy).diff(component) for component in particle.x)
            laws.append(Rate(particle.v, -gradient / particle.m))
        return laws


class PairPotential(Interaction):
    """Base class of pair interactions between particles defined by a potential energy of their distance.

    Subclasses implement `pair_energy`. The force on each particle of a pair is minus the derivative of the energy
    along their separation, derived symbolically.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which pairs do not interact.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.
    """

    over: ClassVar[str] = 'pairs'
    position: ClassVar[str] = 'x'
    first: str = role()
    second: str = role()
    cutoff: float = parameter(default=1.0, domain=Domain.POSITIVE)
    period: float | None = None

    @abstractmethod
    def pair_energy(self, s: Symbols, r: Expression) -> Expression:
        """Return the potential energy of one pair at distance `r`.

        Args:
            s:
                The namespace of the symbols the energy can use.

            r:
                The distance between the two particles.

        Returns:
            energy:
                The pair energy as a symbolic expression.
        """

    def laws(self, s: Symbols) -> list[Law]:
        """Return the velocity changes of both particles of a pair from the derivative of the pair energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        distance = symbol('r', Domain.POSITIVE)
        separation = s.first.x - s.second.x
        r = sqrt(separation.dot_product(separation))
        slope = SR(self.pair_energy(s, distance)).diff(distance).subs({distance: r})
        force = -slope * separation / r
        return [Rate(s.first.v, force / s.first.m), Rate(s.second.v, -force / s.second.m)]
