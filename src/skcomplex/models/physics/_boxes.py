"""Model boxes that confine particles and ideal gases of particle groups."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod

import numpy as np
from sage.all__sagemath_symbolics import SR, sqrt

from ...core import Domain, Expression, VisualSpec, parameter, role, symbol
from ...model import (
    Boundary,
    EuclideanSpace,
    Interaction,
    Law,
    PairInteraction,
    Rate,
    Symbols,
    System,
    either,
    mod,
    where,
)
from ._mechanics import Particle


class ReflectingBox(Interaction):
    """Box `[0, size]` in every dimension whose walls reflect particles elastically.

    A particle that crosses a wall during a step is mirrored back into the box and the velocity component normal to
    the wall changes sign.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or particle group.

        size:
            The side of the box.

    Examples:
        >>> from skcomplex.models.physics import ReflectingBox
        >>> ReflectingBox('box', target='gas', size=2.0).size
        2.0
    """

    target: str = role()
    size: float = parameter(1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reflection of every position and velocity component at the walls.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        laws: list[Law] = []
        for position, velocity in zip(s.target.x, s.target.v, strict=True):
            below, above = position < 0, position > s.size
            mirrored = where(below, -position, where(above, 2 * s.size - position, position))
            laws += [
                Boundary(position, mirrored),
                Boundary(velocity, where(either(below, above), -velocity, velocity)),
            ]
        return laws


class PeriodicBox(Interaction):
    """Box `[0, size)` in every dimension whose opposite sides are joined.

    A particle that leaves through one side reappears at the opposite side with the same velocity.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or particle group.

        size:
            The side of the box.

    Examples:
        >>> from skcomplex.models.physics import PeriodicBox
        >>> PeriodicBox('box', target='gas').size
        1.0
    """

    target: str = role()
    size: float = parameter(1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the wrap of every position component into the box.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Boundary(position, mod(position, s.size)) for position in s.target.x]


class PairPotential(PairInteraction):
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


def ideal_gas(
    n_molecules: int,
    *,
    kT: float = 1.0,
    size: float = 1.0,
    n_dim: int = 2,
    mass: float = 1.0,
    random_state: int | np.random.Generator | None = None,
    label: str = 'gas',
) -> System:
    """Return an ideal gas of non-interacting molecules in a reflecting box.

    Positions are uniform in the box and velocity components are normal with variance `kT / mass`, with the mean
    velocity removed. The molecules are drawn as points colored by speed.

    Args:
        n_molecules:
            The number of molecules.

        kT:
            The temperature in energy units.

        size:
            The side of the box.

        n_dim:
            The number of dimensions.

        mass:
            The mass of every molecule.

        random_state:
            The seed or generator of the initial positions and velocities.

        label:
            The label of the molecule group. The box is labelled `box`.

    Returns:
        system:
            The gas.

    Examples:
        >>> from skcomplex.models.physics import ideal_gas
        >>> ideal_gas(1000, random_state=0).elements[0]
        Group('gas', element=Particle, size=1000)
    """
    rng = np.random.default_rng(random_state)
    velocities = rng.normal(0.0, np.sqrt(kT / mass), size=(n_molecules, n_dim))
    velocities -= velocities.mean(axis=0)
    visual = VisualSpec('points', {'position': 'x', 'radius': size / 200, 'color_by': 'v', 'states': ()})
    molecules = Particle.group(
        label,
        n_molecules,
        x=rng.uniform(0.0, size, size=(n_molecules, n_dim)),
        v=velocities,
        m=mass,
        visual=visual,
    )
    return System(
        elements=[molecules],
        interactions=[ReflectingBox('box', target=label, size=size)],
        space=EuclideanSpace(n_dim),
    )
