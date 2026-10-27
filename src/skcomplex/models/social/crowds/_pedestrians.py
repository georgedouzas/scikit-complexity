"""Pedestrian models: walkers driven by desired velocities and social forces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, exp, oo, vector
from sage.sets.real_set import RealSet

from ....core.elements import GroupElement, ParticleElement
from ....core.interactions import BaseInteraction, Interaction, PairPotentialInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.spaces import ContinuousSpace
from ....core.specs import ParameterSpec, RoleSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import RK4Solver


class MotivationInteraction(Interaction):
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
        >>> from skcomplex.models.social.crowds import MotivationInteraction
        >>> MotivationInteraction('drive', target='walkers', direction=[1.0, 0.0]).speed
        1.34
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.34,
        direction: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.speed = speed
        self.direction = direction
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of the velocity to the desired velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.v, (s.speed * s.direction - s.target.v) / s.tau)]


class WallsInteraction(Interaction):
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
        >>> from skcomplex.models.social.crowds import WallsInteraction
        >>> WallsInteraction('walls', target='walkers', width=5.0).width
        5.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        range: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,  # noqa: A002
        radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.width = width
        self.strength = strength
        self.range = range
        self.radius = radius

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        return [RateLaw(s.target.v, vector(SR, [0, push]))]


class SocialRepulsionInteraction(PairPotentialInteraction):
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
        >>> from skcomplex.models.social.crowds import SocialRepulsionInteraction
        >>> SocialRepulsionInteraction('repulsion', first='walkers', second='walkers').strength
        2.0
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
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        range: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,  # noqa: A002
        radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position, period=period)
        self.strength = strength
        self.range = range
        self.radius = radius

    def _get_pair_energy(self, s: SimpleNamespace, r: Expression) -> Expression:
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
    defaults={'until': 20.0, 'dt': 0.2, 'solver': RK4Solver(dt=0.05)},
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
        GroupElement(
            label,
            element=ParticleElement,
            size=n_walkers,
            values={'x': rng.uniform([0.0, 0.5], [length, length - 0.5], (n_walkers, 2))},
        )
        for label in ('rightward', 'leftward')
    ]
    interactions: list[BaseInteraction] = []
    for label, sign in (('rightward', 1.0), ('leftward', -1.0)):
        interactions += [
            MotivationInteraction(f'{label}_drive', target=label, speed=speed, direction=[sign, 0.0]),
            WallsInteraction(f'{label}_walls', target=label, width=length),
        ]
    pairs = (('rightward', 'rightward'), ('leftward', 'leftward'), ('rightward', 'leftward'))
    interactions += [
        SocialRepulsionInteraction(f'{first}_{second}', first=first, second=second, cutoff=cutoff, period=length)
        for first, second in pairs
    ]
    return System(
        elements=walkers,
        interactions=interactions,
        space=ContinuousSpace(extent=(length, length), boundary='periodic'),
    )
