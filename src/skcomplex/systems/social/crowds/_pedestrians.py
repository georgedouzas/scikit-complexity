"""Pedestrian systems: walkers driven by desired velocities and social forces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, exp, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import ObjectiveLaw, RateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.components.types import Target
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.elements import PointElement
from ...general.gradients import GradientInteraction


class MotivationInteraction(Interaction):
    """Driving force of the social force system, which relaxes a walker's velocity to its desired velocity.

    The force per unit mass is `(speed * direction - x') / tau`, where the velocity `x'` is the derivative of the
    position `x`.

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
        >>> from skcomplex.systems.social.crowds import MotivationInteraction
        >>> MotivationInteraction('drive', target='walkers', direction=[1.0, 0.0]).speed
        1.34
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 1.34,
        direction: npt.ArrayLike = 0.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.speed = speed
        self.direction = direction
        self.tau = tau

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the relaxation of the velocity of the target towards its desired velocity.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x = s.target.x
            self.laws_ = [RateLaw(x.diff(t, 2), (s.speed * s.direction - x.diff(t)) / s.tau)]


class SocialRepulsionInteraction(Interaction):
    """Repulsion between two walkers of the social force system.

    Its force is `strength * exp((2 * radius - r) / falloff)` along the line between the walkers. It derives from the
    pair energy `strength * falloff * exp((2 * radius - r) / falloff)`, and it runs once for every pair of walkers of
    the groups `first` and `second` closer than the cutoff, pushing both apart.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group of the first walkers.

        second:
            The path of the group of the second walkers, which may be the first group.

        cutoff:
            The distance beyond which walkers do not interact.

        strength:
            The strength of the repulsion.

        falloff:
            The distance over which the repulsion falls off.

        radius:
            The radius of a walker.

    Examples:
        >>> from skcomplex.systems.social.crowds import SocialRepulsionInteraction
        >>> SocialRepulsionInteraction('repulsion', first='walkers', second='walkers').strength
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        cutoff: float = 1.0,
        strength: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        falloff: Annotated[float, RealSet.open(0, oo)] = 0.3,
        radius: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.cutoff = cutoff
        self.strength = strength
        self.falloff = falloff
        self.radius = radius

    def _get_within(self) -> RadiusNeighborhood:
        """Return the neighbourhood of the cutoff, whose pairs of walkers repel."""
        return RadiusNeighborhood(self.cutoff)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the energy of a pair of walkers, which pushes both apart along the line between them.

        Args:
            context:
                The namespace of the pair, `s.first` and `s.second`, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            separation = s.first.x - s.second.x
            r = sqrt(separation.dot_product(separation))
            self.laws_ = [ObjectiveLaw(s.strength * s.falloff * exp((2 * s.radius - r) / s.falloff))]


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
        >>> from skcomplex.systems.social.crowds import WallsInteraction
        >>> WallsInteraction('walls', target='walkers', width=5.0).width
        5.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        width: Annotated[float, RealSet.open(0, oo)] = 10.0,
        strength: Annotated[float, RealSet.closed_open(0, oo)] = 10.0,
        falloff: Annotated[float, RealSet.open(0, oo)] = 0.2,
        radius: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
    ) -> None:
        super().__init__(label, target=target, potential=self._get_potential)
        self.width = width
        self.strength = strength
        self.falloff = falloff
        self.radius = radius

    def _get_rate_law(self, state: Target, rate: object, member: SimpleNamespace, time: Expression) -> RateLaw:
        """Return the force per unit mass of the walls on the acceleration of a walker."""
        return RateLaw(state.diff(time, 2), rate)

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return the potential of both walls at the position of a walker."""
        y = s.target.x[1]
        return s.strength * s.falloff * (exp((s.radius - y) / s.falloff) + exp((s.radius - s.width + y) / s.falloff))


class SocialForceSystem(BaseSystem):
    """Two crowds walking in opposite directions along a periodic corridor.

    The crowds form lanes. The system has the groups `rightward` and `leftward`.

    Args:
        n_walkers:
            The number of walkers in each direction.

        length:
            The length and the width of the corridor, periodic along its length.

        speed:
            The desired speed.

        random_state:
            The seed or generator of the initial positions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.crowds import SocialForceSystem
        >>> [group.size for group in SocialForceSystem(5).prepare().system_.elements]
        [5, 5]
    """

    field = 'social/crowds'
    topics = ('pedestrians', 'particles', 'lanes', 'self-organization')
    references = ('Helbing and Molnár 1995', 'Helbing 2001')
    until = 20.0
    dt = 0.2

    def __init__(
        self,
        n_walkers: int = 20,
        *,
        length: float = 10.0,
        speed: float = 1.34,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.length = length
        self.speed = speed
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        cutoff = 2.0
        corridor = ContinuousSpace(extent=(self.length, self.length), boundary='periodic')
        walkers = [
            PointElement(
                label,
                x=rng.uniform([0.0, 0.5], [self.length, self.length - 0.5], (self.n_walkers, 2)),
                size=self.n_walkers,
                space=corridor,
            )
            for label in ('rightward', 'leftward')
        ]
        interactions: list[BaseInteraction] = []
        for label, sign in (('rightward', 1.0), ('leftward', -1.0)):
            interactions += [
                MotivationInteraction(f'{label}_drive', target=label, speed=self.speed, direction=[sign, 0.0]),
                WallsInteraction(f'{label}_walls', target=label, width=self.length),
            ]
        pairs = (('rightward', 'rightward'), ('leftward', 'leftward'), ('rightward', 'leftward'))
        interactions += [
            SocialRepulsionInteraction(f'{first}_{second}', first=first, second=second, cutoff=cutoff)
            for first, second in pairs
        ]
        self.system_ = System(
            self.label,
            elements=walkers,
            interactions=interactions,
            space=corridor,
            solver=RK4Solver(dt=0.05),
        )
