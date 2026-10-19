"""String models: chains of masses on springs, guitar strings, and standing waves on a continuous string."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import sin

from ....core.base import Domain, Expression, parameter, role, vector
from ....core.model import (
    Element,
    EuclideanSpace,
    Field,
    LatticeSpace,
    Particle,
    Potential,
    Rate,
    Spring,
    Symbols,
    System,
    laplacian,
)
from ....core.runtime import register_model
from ....core.solvers import VelocityVerlet

WaveString = Element.define(
    'WaveString',
    states={'u': 0.0, 'v': 0.0},
    parameters={'c': parameter(default=1.0, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.u, s.v), Rate(s.v, s.c**2 * laplacian(s.u))],
)


def _pluck(positions: np.ndarray, length: float, height: float, at: float) -> np.ndarray:
    """Return the triangular shape of a string pulled to a height at a fraction of its length."""
    peak = at * length
    return np.where(positions <= peak, height * positions / peak, height * (length - positions) / (length - peak))


class DrivenAnchor(Potential):
    """Spring that ties a particle to an anchor point, which oscillates transversally as `amplitude sin(omega t)`.

    The energy is `k |x - center - (0, amplitude sin(omega t))|² / 2`, and a zero amplitude gives a fixed anchor.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The stiffness of the spring.

        center:
            The rest position of the anchor.

        amplitude:
            The amplitude of the transverse motion of the anchor.

        omega:
            The angular frequency of the anchor.

    Examples:
        >>> from skcomplex.models.physics.waves import DrivenAnchor
        >>> DrivenAnchor('end', target='mass_0', center=[0.0, 0.0], amplitude=0.1).amplitude
        0.1
    """

    target: str = role()
    k: float = parameter(default=1.0, domain=Domain.POSITIVE)
    center: list[float] = parameter(default=0.0, vector=True)
    amplitude: float = parameter(default=0.0)
    omega: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def energy(self, s: Symbols) -> Expression:
        """Return `k |x - center - (0, amplitude sin(omega t))|² / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        anchor = s.center + vector([0, s.amplitude * sin(s.omega * s.t)])
        displacement = s.target.x - anchor
        return s.k * displacement.dot_product(displacement) / 2


@register_model(
    field='physics/waves',
    tags=('waves', 'ode', 'springs', 'normal modes', 'mechanics'),
    references=('Argun et al. 2021',),
    defaults={'until': 20.0, 'dt': 0.05, 'solver': VelocityVerlet(dt=0.01)},
)
def spring_chain(
    n_masses: int = 15,
    *,
    length: float = 1.0,
    k: float = 1.0,
    mass: float = 0.01,
    pluck: float = 0.05,
    pluck_at: float = 0.5,
    amplitude: float = 0.0,
    omega: float = 2.0,
) -> System:
    """Return masses joined in a row by springs of zero rest length between two anchors, a discrete string.

    The masses start evenly spaced between the anchors at `0` and `length`, stretched, so the chain carries tension.
    With `pluck`, the masses start on a triangle of that height peaking at the fraction `pluck_at` of the length, the
    pluck of a guitar string. With `amplitude`, the left anchor moves transversally as `amplitude sin(omega t)` and
    sends a transverse wave along the chain. Springs of zero rest length are linear, so the transverse modes are
    independent, with frequencies `2 sqrt(k / m) sin(n pi / (2 (N + 1)))`.

    Args:
        n_masses:
            The number of masses between the anchors.

        length:
            The distance between the anchors.

        k:
            The stiffness of every spring.

        mass:
            The mass of every mass.

        pluck:
            The initial transverse displacement of the plucked point.

        pluck_at:
            The plucked point, as a fraction of the length.

        amplitude:
            The amplitude of the transverse motion of the left anchor.

        omega:
            The angular frequency of the left anchor.

    Returns:
        system:
            The model, with the masses `mass_<i>` and their positions `mass_<i>.x` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics.waves import spring_chain
        >>> len(spring_chain(4).interactions)
        5
    """
    spacing = length / (n_masses + 1)
    positions = spacing * np.arange(1, n_masses + 1)
    heights = _pluck(positions, length, pluck, pluck_at) if pluck else np.zeros(n_masses)
    masses = [
        Particle(f'mass_{index}', x=[float(positions[index]), float(heights[index])], v=[0.0, 0.0], m=mass)
        for index in range(n_masses)
    ]
    springs = [
        Spring(f'spring_{index}', first=f'mass_{index}', second=f'mass_{index + 1}', k=k)
        for index in range(n_masses - 1)
    ]
    anchors = [
        DrivenAnchor('left_end', target='mass_0', k=k, center=[0.0, 0.0], amplitude=amplitude, omega=omega),
        DrivenAnchor('right_end', target=f'mass_{n_masses - 1}', k=k, center=[length, 0.0]),
    ]
    return System(elements=masses, interactions=[*anchors, *springs], space=EuclideanSpace(2))


@register_model(
    field='physics/waves',
    tags=('waves', 'field', 'pde', 'normal modes'),
    references=('Starkovich 2021',),
    defaults={'until': 4.0, 'dt': 0.05},
)
def standing_waves(
    n_sites: int = 99,
    *,
    length: float = 2.0,
    c: float = 1.0,
    mode: int = 1,
    amplitude: float = 1.0,
) -> System:
    """Return a string with fixed ends on `[-length / 2, length / 2]` that vibrates as a standing wave.

    The displacement follows the wave equation `u'' = c² u_xx`, written as `u' = v` and `v' = c² laplacian(u)` on a
    lattice whose edges hold the string at zero. The string starts at rest in the mode
    `amplitude sin(mode pi (x + length / 2) / length)`, which oscillates in place with the period
    `2 length / (mode c)`, up to the discretization.

    Args:
        n_sites:
            The number of sites inside the string.

        length:
            The length of the string.

        c:
            The speed of the waves.

        mode:
            The number of half wavelengths along the string.

        amplitude:
            The initial amplitude.

    Returns:
        system:
            The model, with the displacement `string.u` and the velocity `string.v`.

    Examples:
        >>> from skcomplex.models.physics.waves import standing_waves
        >>> standing_waves(9).elements[0].size
        9
    """
    spacing = length / (n_sites + 1)
    positions = spacing * np.arange(1, n_sites + 1)
    shape = amplitude * np.sin(mode * math.pi * positions / length)
    space = LatticeSpace(n_sites, spacing=spacing)
    edges = ('fixed', 0.0)
    string = Field('string', WaveString, space, boundary={'u': edges, 'v': edges}, values={'u': shape, 'c': c})
    return System(elements=[string])
