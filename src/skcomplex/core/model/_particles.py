"""Model point particles, forces, potentials, pair potentials, boxes, and thermostats."""

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
    SymbolicVector,
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
from ._laws import Law, Noise, Rate
from ._symbols import Symbols


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
        >>> from skcomplex.core.model import ConstantForce
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
        >>> from skcomplex.core.model import LinearDrag
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
        >>> from skcomplex.core.model import HarmonicPotential
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
        >>> from skcomplex.core.model import Spring
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
        >>> from skcomplex.core.model import Gravity
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


class LennardJones(PairPotential):
    """Lennard-Jones pair potential, `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`, without a shift.

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

        epsilon:
            The depth of the potential well.

        sigma:
            The distance at which the potential is zero.

    Examples:
        >>> from skcomplex.core.model import LennardJones
        >>> LennardJones('lj', first='gas', second='gas', cutoff=2.5).sigma
        1.0
    """

    epsilon: float = parameter(default=1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s: Symbols, r: Expression) -> Expression:
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`.

        Args:
            s:
                The namespace of the symbols the energy can use.

            r:
                The distance between the two particles.

        Returns:
            energy:
                The pair energy as a symbolic expression.
        """
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


class SoftSphere(PairPotential):
    """Soft repulsion between overlapping spheres, `k * (cutoff - r) ** 2 / 2` closer than the cutoff.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The contact distance, the sum of the two radii.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        k:
            The stiffness.

    Examples:
        >>> from skcomplex.core.model import SoftSphere
        >>> SoftSphere('contact', first='gas', second='gas', cutoff=0.1, k=50.0).k
        50.0
    """

    k: float = parameter(default=100.0, domain=Domain.POSITIVE)

    def pair_energy(self, s: Symbols, r: Expression) -> Expression:
        """Return `k * (cutoff - r) ** 2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

            r:
                The distance between the two particles.

        Returns:
            energy:
                The pair energy as a symbolic expression.
        """
        return s.k * (s.cutoff - r) ** 2 / 2


class LangevinThermostat(Interaction):
    """Langevin thermostat, which adds a drag `-gamma * v` and a random force that hold particles at a temperature.

    Every velocity component follows `dv = -gamma * v / m * dt + sqrt(2 * gamma * temperature) / m * dW`, with the
    temperature in energy units, so a system with a thermostat is stochastic and its velocities relax to the
    Maxwell-Boltzmann distribution at that temperature.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or particle group.

        temperature:
            The temperature in energy units.

        gamma:
            The friction coefficient.

    Examples:
        >>> from skcomplex.core.model import LangevinThermostat
        >>> LangevinThermostat('bath', target='gas', temperature=2.0).temperature
        2.0
    """

    target: str = role()
    temperature: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    gamma: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drag and the random force on every velocity component.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        velocity, mass = s.target.v, s.target.m
        return [
            Rate(velocity, -s.gamma * velocity / mass),
            Noise(velocity, sqrt(2 * s.gamma * s.temperature) / mass),
        ]
