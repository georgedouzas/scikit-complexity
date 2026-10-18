"""Model standard pair potentials and thermostats for particle groups."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sympy

from ...core import Domain, parameter, role
from ...model import Interaction, Law, Noise, Rate, Symbols
from ._boxes import PairPotential


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
        >>> from skcomplex.models.physics import LennardJones
        >>> LennardJones('lj', first='gas', second='gas', cutoff=2.5).sigma
        1.0
    """

    epsilon: float = parameter(1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s: Symbols, r: sympy.Symbol) -> sympy.Expr:
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
        >>> from skcomplex.models.physics import SoftSphere
        >>> SoftSphere('contact', first='gas', second='gas', cutoff=0.1, k=50.0).k
        50.0
    """

    k: float = parameter(100.0, domain=Domain.POSITIVE)

    def pair_energy(self, s: Symbols, r: sympy.Symbol) -> sympy.Expr:
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
        >>> from skcomplex.models.physics import LangevinThermostat
        >>> LangevinThermostat('bath', target='gas', temperature=2.0).temperature
        2.0
    """

    target: str = role()
    temperature: float = parameter(1.0, domain=Domain.NONNEGATIVE)
    gamma: float = parameter(1.0, domain=Domain.POSITIVE)

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
            Noise(velocity, sympy.sqrt(2 * s.gamma * s.temperature) / mass),
        ]
