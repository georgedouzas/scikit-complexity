"""Pedestrian models: walkers driven by desired velocities and social forces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import GroupElement, PointElement
from ....core.interactions import (
    BaseInteraction,
    GradientInteraction,
    PairCouplingInteraction,
    RelaxationInteraction,
)
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....specs import ParameterSpec, RoleSpec
from ..._catalog import register_model


class MotivationInteraction(RelaxationInteraction):
    """Driving force of the social force model, which relaxes a walker's velocity to its desired velocity.

    The force per unit mass is `(speed * direction - x_d1) / tau`, where the velocity `x_d1` is the first derivative of
    the position `x`.

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
        super().__init__(label, target=target, state='x_d1')
        self.speed = speed
        self.direction = direction
        self.tau = tau

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the desired velocity."""
        return namespace.speed * namespace.direction

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time."""
        return namespace.tau


class WallsInteraction(GradientInteraction):
    """Repulsion of walkers by two walls of a corridor at `y = 0` and `y = width`.

    It is a `GradientInteraction` of the position `x` down the potential
    `strength falloff (exp((radius - y) / falloff) + exp((radius - width + y) / falloff))`, so the force per unit mass
    from a wall at distance `d` is `strength * exp((radius - d) / falloff)`, pointing away from the wall.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walker or group of walkers.

        width:
            The distance between the walls.

        strength:
            The strength of the repulsion.

        falloff:
            The distance over which the repulsion falls off.

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
        falloff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, target=target, potential=self._get_potential)
        self.width = width
        self.strength = strength
        self.falloff = falloff
        self.radius = radius

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return the potential of both walls at the position of a walker."""
        y = s.target.x[1]
        return s.strength * s.falloff * (exp((s.radius - y) / s.falloff) + exp((s.radius - s.width + y) / s.falloff))


class SocialRepulsionInteraction(PairCouplingInteraction):
    """Repulsion between two walkers, the force `strength * exp((2 * radius - r) / falloff)` of the social force model.

    It derives from the pair energy `strength * falloff * exp((2 * radius - r) / falloff)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or walker of the first members.

        second:
            The path of the group or walker of the second members.

        cutoff:
            The distance beyond which walkers do not interact.

        position:
            The name of the vector state that holds the positions of the walkers.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        strength:
            The strength of the repulsion.

        falloff:
            The distance over which the repulsion falls off.

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
        falloff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            potential=self._get_pair_energy,
            period=period,
        )
        self.strength = strength
        self.falloff = falloff
        self.radius = radius

    @staticmethod
    def _get_pair_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return the pair energy `strength * falloff * exp((2 * radius - r) / falloff)`."""
        return s.strength * s.falloff * exp((2 * s.radius - r) / s.falloff)


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
            element=PointElement,
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
