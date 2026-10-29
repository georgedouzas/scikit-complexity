"""Named forces, potentials, pair potentials, and thermostats of particles, with their native LAMMPS translations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from .....core.interactions import BaseInteraction, Interaction, PairPotentialInteraction, PotentialInteraction
from .....core.laws import BaseLaw, NoiseLaw, RateLaw
from .....engine.types import PairArguments
from .....specs import LammpsFixSpec, ParameterSpec, RoleSpec
from .....types import Array


def _distance(first: FreeModuleElement, second: FreeModuleElement) -> Expression:
    """Return the Euclidean distance between two position vectors."""
    difference = first - second
    return sqrt(difference.dot_product(difference))


class ConstantForceInteraction(Interaction):
    """Constant force on a particle.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        F:
            The force vector. A number applies to every component.

    Examples:
        >>> from skcomplex.models.physics.common.particles import ConstantForceInteraction
        >>> ConstantForceInteraction('gravity', target='ball', F=[0, -9.81]).F
        [0, -9.81]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        F: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.F = F

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the velocity change `F / m`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.v, s.F / s.target.m)]


class LinearDragInteraction(Interaction):
    """Drag force proportional and opposite to the velocity of a particle.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        gamma:
            The drag coefficient.

    Examples:
        >>> from skcomplex.models.physics.common.particles import LinearDragInteraction
        >>> LinearDragInteraction('air', target='ball', gamma=0.1).gamma
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the velocity change `-gamma * v / m`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.v, -s.gamma * s.target.v / s.target.m)]


class HarmonicPotentialInteraction(PotentialInteraction):
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
        >>> from skcomplex.models.physics.common.particles import HarmonicPotentialInteraction
        >>> HarmonicPotentialInteraction('trap', target='ball', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.center = center

    def _get_energy(self, s: SimpleNamespace) -> Expression:
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


class SpringInteraction(PotentialInteraction):
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
        >>> from skcomplex.models.physics.common.particles import SpringInteraction
        >>> SpringInteraction('spring', first='a', second='b', k=3.0, length=1.0).length
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        length: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.k = k
        self.length = length

    def _get_energy(self, s: SimpleNamespace) -> Expression:
        """Return `k * (|x_first - x_second| - length)**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        return s.k * (_distance(s.first.x, s.second.x) - s.length) ** 2 / 2


class GravityInteraction(PotentialInteraction):
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
        >>> from skcomplex.models.physics.common.particles import GravityInteraction
        >>> GravityInteraction('gravity', first='sun', second='earth', G=1.0).G
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        G: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.G = G

    def _get_energy(self, s: SimpleNamespace) -> Expression:
        """Return `-G * m_first * m_second / |x_first - x_second|`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        return -s.G * s.first.m * s.second.m / _distance(s.first.x, s.second.x)


class LennardJonesInteraction(PairPotentialInteraction):
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

        position:
            The name of the vector state that holds the positions of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        epsilon:
            The depth of the potential well.

        sigma:
            The distance at which the potential is zero.

    Examples:
        >>> from skcomplex.models.physics.common.particles import LennardJonesInteraction
        >>> LennardJonesInteraction('lj', first='gas', second='gas', cutoff=2.5).sigma
        1.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position, period=period)
        self.epsilon = epsilon
        self.sigma = sigma

    def _get_pair_energy(self, s: SimpleNamespace, r: Expression) -> Expression:
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


class SoftSphereInteraction(PairPotentialInteraction):
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

        position:
            The name of the vector state that holds the positions of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        k:
            The stiffness.

    Examples:
        >>> from skcomplex.models.physics.common.particles import SoftSphereInteraction
        >>> SoftSphereInteraction('contact', first='gas', second='gas', cutoff=0.1, k=50.0).k
        50.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position, period=period)
        self.k = k

    def _get_pair_energy(self, s: SimpleNamespace, r: Expression) -> Expression:
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


class LangevinThermostatInteraction(Interaction):
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
        >>> from skcomplex.models.physics.common.particles import LangevinThermostatInteraction
        >>> LangevinThermostatInteraction('bath', target='gas', temperature=2.0).temperature
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.temperature = temperature
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(velocity, -s.gamma * velocity / mass),
            NoiseLaw(velocity, sqrt(2 * s.gamma * s.temperature) / mass),
        ]


def _lennard_jones(value: Callable[[str], Array]) -> str:
    """Return the coefficients of `lj/cut`."""
    return f'{float(value("epsilon")[0])} {float(value("sigma")[0])}'


def _addforce(fix: LammpsFixSpec) -> str:
    """Return the `fix addforce` command of a constant force."""
    force = [*map(float, fix.value('F')), 0.0, 0.0, 0.0]
    return f'fix {fix.name} {fix.group} addforce {force[0]} {force[1]} {force[2]}'


def _viscous(fix: LammpsFixSpec) -> str:
    """Return the `fix viscous` command of a linear drag."""
    return f'fix {fix.name} {fix.group} viscous {float(fix.value("gamma")[0])}'


def _langevin(fix: LammpsFixSpec) -> str:
    """Return the `fix langevin` command of a Langevin thermostat."""
    temperature, gamma = float(fix.value('temperature')[0]), float(fix.value('gamma')[0])
    return f'fix {fix.name} {fix.group} langevin {temperature} {temperature} {fix.mass() / gamma} {fix.seed()}'


LAMMPS_PAIRS: dict[type[BaseInteraction], tuple[str, PairArguments]] = {
    LennardJonesInteraction: ('lj/cut', _lennard_jones),
}
LAMMPS_FIXES: dict[type[BaseInteraction], Callable[[LammpsFixSpec], str]] = {
    ConstantForceInteraction: _addforce,
    LinearDragInteraction: _viscous,
    LangevinThermostatInteraction: _langevin,
}
