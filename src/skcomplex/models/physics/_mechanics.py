"""Model point particles under forces and potentials in Euclidean space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from typing import ClassVar

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, sqrt

from ...core.base import (
    Domain,
    Expression,
    SymbolicVector,
    VariableKind,
    VisualSpec,
    declarations_of,
    parameter,
    role,
    state,
    vector,
)
from ...core.model import Element, Interaction, Law, Rate, Symbols


def _distance(first: SymbolicVector, second: SymbolicVector) -> Expression:
    """Return the Euclidean distance between two position vectors."""
    difference = first - second
    return sqrt(difference.dot_product(difference))


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
        >>> from skcomplex.models.physics import Particle
        >>> Particle('ball', x=[0, 1], m=2.0)
        Particle(label='ball', m=2.0, x=[0, 1])
    """

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x'})
    x: npt.ArrayLike = state(default=0.0, vector=True)
    v: npt.ArrayLike = state(default=0.0, vector=True)
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


class ConstantForce(Interaction):
    """Constant force on a particle.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        F:
            The force vector. A number applies to every component.

    Examples:
        >>> from skcomplex.models.physics import ConstantForce
        >>> ConstantForce('gravity', target='ball', F=[0, -9.81]).F
        [0, -9.81]
    """

    target: str = role()
    F: npt.ArrayLike = parameter(default=0.0, vector=True)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the velocity change `F / m`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.v, s.F / s.target.m)]


class LinearDrag(Interaction):
    """Drag force proportional and opposite to the velocity of a particle.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        gamma:
            The drag coefficient.

    Examples:
        >>> from skcomplex.models.physics import LinearDrag
        >>> LinearDrag('air', target='ball', gamma=0.1).gamma
        0.1
    """

    target: str = role()
    gamma: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the velocity change `-gamma * v / m`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.v, -s.gamma * s.target.v / s.target.m)]


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


class HarmonicPotential(Potential):
    """Harmonic potential that pulls a particle towards a center.

    The energy is `k * |x - center|**2 / 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The stiffness.

        center:
            The center. A number applies to every component.

    Examples:
        >>> from skcomplex.models.physics import HarmonicPotential
        >>> HarmonicPotential('trap', target='ball', k=2.0).k
        2.0
    """

    target: str = role()
    k: float = parameter(default=1.0, domain=Domain.POSITIVE)
    center: npt.ArrayLike = parameter(default=0.0, vector=True)

    def energy(self, s: Symbols) -> Expression:
        """Return `k * |x - center|**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        displacement = s.target.x - s.center
        return s.k * displacement.dot_product(displacement) / 2


class Spring(Potential):
    """Spring between two particles.

    The energy is `k * (|x_first - x_second| - length)**2 / 2`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        k:
            The stiffness.

        length:
            The rest length.

    Examples:
        >>> from skcomplex.models.physics import Spring
        >>> Spring('spring', first='a', second='b', k=3.0, length=1.0).length
        1.0
    """

    first: str = role()
    second: str = role()
    k: float = parameter(default=1.0, domain=Domain.POSITIVE)
    length: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def energy(self, s: Symbols) -> Expression:
        """Return `k * (|x_first - x_second| - length)**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        return s.k * (_distance(s.first.x, s.second.x) - s.length) ** 2 / 2


class Gravity(Potential):
    """Newtonian gravity between two particles.

    The energy is `-G * m_first * m_second / |x_first - x_second|`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        G:
            The gravitational constant.

    Examples:
        >>> from skcomplex.models.physics import Gravity
        >>> Gravity('gravity', first='sun', second='earth', G=1.0).G
        1.0
    """

    first: str = role()
    second: str = role()
    G: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def energy(self, s: Symbols) -> Expression:
        """Return `-G * m_first * m_second / |x_first - x_second|`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        return -s.G * s.first.m * s.second.m / _distance(s.first.x, s.second.x)
