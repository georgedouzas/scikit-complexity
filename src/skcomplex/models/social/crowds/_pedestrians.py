"""Pedestrian models: walkers driven by desired velocities and social forces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, role, vector
from ....core.components import Particle
from ....core.interactions import Interaction, PairPotential
from ....core.laws import Law, Rate, Symbols
from ....core.runtime import register_model
from ....core.solvers import RK4
from ....core.space import ContinuousSpace
from ....core.system import System


class Motivation(Interaction):
    """Driving force of the social force model, which relaxes a walker's velocity to its desired velocity.

    The force per unit mass is `(speed * direction - v) / tau`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walker or group of walkers.

        speed:
            The desired speed.

        direction:
            The unit vector of the desired direction.

        tau:
            The relaxation time.

    Examples:
        >>> from skcomplex.models.social.crowds import Motivation
        >>> Motivation('drive', target='walkers', direction=[1.0, 0.0]).speed
        1.34
    """

    target: str = role()
    speed: float = parameter(default=1.34, domain=Domain.NONNEGATIVE)
    direction: npt.ArrayLike = parameter(default=0.0, vector=True)
    tau: float = parameter(default=0.5, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the relaxation of the velocity to the desired velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.v, (s.speed * s.direction - s.target.v) / s.tau)]


class Walls(Interaction):
    """Repulsion of walkers by two walls of a corridor at `y = 0` and `y = width`.

    The force per unit mass from a wall at distance `d` is `strength * exp((radius - d) / range)`, pointing away from
    the wall.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walker or group of walkers.

        width:
            The distance between the walls.

        strength:
            The strength of the repulsion.

        range:
            The range of the repulsion.

        radius:
            The radius of a walker.

    Examples:
        >>> from skcomplex.models.social.crowds import Walls
        >>> Walls('walls', target='walkers', width=5.0).width
        5.0
    """

    target: str = role()
    width: float = parameter(default=10.0, domain=Domain.POSITIVE)
    strength: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    range: float = parameter(default=0.2, domain=Domain.POSITIVE)
    radius: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the repulsion of both walls.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        y = s.target.x[1]
        push = s.strength * (exp((s.radius - y) / s.range) - exp((s.radius - s.width + y) / s.range))
        return [Rate(s.target.v, vector([0, push]))]


class SocialRepulsion(PairPotential):
    """Repulsion between two walkers, the force `strength * exp((2 * radius - r) / range)` of the social force model.

    It derives from the pair energy `strength * range * exp((2 * radius - r) / range)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or walker of the first members.

        second:
            The path of the group or walker of the second members.

        cutoff:
            The distance beyond which walkers do not interact.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        strength:
            The strength of the repulsion.

        range:
            The range of the repulsion.

        radius:
            The radius of a walker.

    Examples:
        >>> from skcomplex.models.social.crowds import SocialRepulsion
        >>> SocialRepulsion('repulsion', first='walkers', second='walkers').strength
        2.0
    """

    strength: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    range: float = parameter(default=0.3, domain=Domain.POSITIVE)
    radius: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)

    def pair_energy(self, s: Symbols, r: Expression) -> Expression:
        """Return the pair energy `strength * range * exp((2 * radius - r) / range)`.

        Args:
            s:
                The namespace of the symbols the energy can use.

            r:
                The distance between the two walkers.

        Returns:
            energy:
                The pair energy as a symbolic expression.
        """
        return s.strength * s.range * exp((2 * s.radius - r) / s.range)


@register_model(
    field='social/crowds',
    tags=('pedestrians', 'particles', 'lanes', 'self-organization'),
    references=('Helbing and Molnár 1995', 'Helbing 2001'),
    defaults={'until': 20.0, 'dt': 0.2, 'solver': RK4(dt=0.05)},
)
def social_force(
    n_walkers: int = 20,
    *,
    length: float = 10.0,
    speed: float = 1.34,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return two crowds walking in opposite directions along a periodic corridor, which form lanes.

    Args:
        n_walkers:
            The number of walkers in each direction.

        length:
            The length and the width of the corridor, periodic along its length.

        speed:
            The desired speed.

        random_state:
            The seed or generator of the initial positions.

    Returns:
        system:
            The model, with the groups `rightward` and `leftward`.

    Examples:
        >>> from skcomplex.models.social.crowds import social_force
        >>> [group.size for group in social_force(5).elements]
        [5, 5]
    """
    rng = np.random.default_rng(random_state)
    cutoff = 2.0
    walkers = [
        Particle.group(label, n_walkers, x=rng.uniform([0.0, 0.5], [length, length - 0.5], (n_walkers, 2)))
        for label in ('rightward', 'leftward')
    ]
    interactions: list[Interaction] = []
    for label, sign in (('rightward', 1.0), ('leftward', -1.0)):
        interactions += [
            Motivation(f'{label}_drive', target=label, speed=speed, direction=[sign, 0.0]),
            Walls(f'{label}_walls', target=label, width=length),
        ]
    pairs = (('rightward', 'rightward'), ('leftward', 'leftward'), ('rightward', 'leftward'))
    interactions += [
        SocialRepulsion(f'{first}_{second}', first=first, second=second, cutoff=cutoff, period=length)
        for first, second in pairs
    ]
    return System(
        components=walkers,
        interactions=interactions,
        space=ContinuousSpace(length, length, boundary='periodic'),
    )
