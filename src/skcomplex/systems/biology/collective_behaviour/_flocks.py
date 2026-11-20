"""Flocks and swarms that move together."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import SR, exp, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.agents import MovementInteraction
from ...general.elements import PointElement
from ._zones import ClampSpec, RadiusSpec, ZoneCouplingInteraction


class BoidView(AgentViewProtocol, Protocol):
    """View of a boid, with its states and parameters as attributes.

    The rules of `BoidElement` read them with checked types.
    """

    radius: float
    separation_radius: float
    cohesion: float
    separation: float
    alignment: float
    min_speed: float
    max_speed: float
    width: float
    height: float


class BoidElement(Element):
    """Boid of Reynolds, a bird that steers by three rules.

    The rules steer it toward, away from, and along its neighbours.

    Every step a boid sees the boids within `radius`. `FlockingInteraction` steers it by cohesion toward their center,
    by separation away from those closer than `separation_radius`, and by alignment toward their mean velocity, with
    its speed between `min_speed` and `max_speed`, and `FlightInteraction` moves it on a torus of side `width` by
    `height`. Its position is `x` and `y`, and its velocity `"x'"` and `"y'"`, which its rules read and change as
    `agent.get_value("x'")` and `agent.set_value("x'", value)`.

    Args:
        label:
            The label of the agent or group.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        radius:
            The distance within which a boid sees others.

        separation_radius:
            The distance within which a boid avoids others.

        cohesion:
            The weight of cohesion.

        separation:
            The weight of separation.

        alignment:
            The weight of alignment.

        min_speed:
            The smallest speed.

        max_speed:
            The largest speed.

        width:
            The width of the torus.

        height:
            The height of the torus.

        values:
            Further values by name, such as the velocity `{"x'": 1.0, "y'": 0.0}`.

        size:
            The number of boids, or `None` for one.

        space:
            The space of the boids, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BoidElement
        >>> BoidElement('flock', x=5.0, y=5.0, values={"x'": 1.0, "y'": 0.0}, size=10).size
        10
    """

    visual = VisualSpec('agents', {'color_by': "x'"})

    def __init__(
        self,
        label: str,
        *,
        x: float | Array = 0.0,
        y: float | Array = 0.0,
        radius: Annotated[float, RealSet.open(0, oo)] = 5.0,
        separation_radius: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        cohesion: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        separation: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        alignment: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        min_speed: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        max_speed: Annotated[float, RealSet.open(0, oo)] = 1.0,
        width: Annotated[float, RealSet.open(0, oo)] = 30.0,
        height: Annotated[float, RealSet.open(0, oo)] = 30.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x
        self.y = y
        self.radius = radius
        self.separation_radius = separation_radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.width = width
        self.height = height


class SpatialBoidView(AgentViewProtocol, Protocol):
    """View of a boid in three dimensions.

    The view has the states and parameters of the boid as attributes.

    The rules of `SpatialBoidElement` read them with checked types.
    """

    radius: float
    separation_radius: float
    cohesion: float
    separation: float
    alignment: float
    min_speed: float
    max_speed: float


class SpatialBoidElement(Element):
    """Boid in three dimensions, a bird that steers by the boids it sees.

    It steers by the velocity and the position of those boids.

    Every step `HeadingInteraction` steers a boid by cohesion, separation, and alignment with the boids within `radius`,
    with its speed between `min_speed` and `max_speed`, and `SpatialFlightInteraction` moves it along its velocity. Its
    position is `x`, `y`, and `z`, and its velocity `"x'"`, `"y'"`, and `"z'"`, which its rules read and
    change as `agent.get_value("x'")` and `agent.set_value("x'", value)`.

    Args:
        label:
            The label of the agent or group.

        x:
            The initial position along the first axis.

        y:
            The initial position along the second axis.

        z:
            The initial position along the third axis.

        radius:
            The distance within which a boid sees others.

        separation_radius:
            The distance within which a boid avoids others.

        cohesion:
            The weight of cohesion.

        separation:
            The weight of separation.

        alignment:
            The weight of alignment.

        min_speed:
            The smallest speed.

        max_speed:
            The largest speed.

        values:
            Further values by name, such as the velocity `{"x'": 1.0, "y'": 0.0, "z'": 0.0}`.

        size:
            The number of boids, or `None` for one.

        space:
            The space of the boids, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SpatialBoidElement
        >>> SpatialBoidElement('flock', size=10).size
        10
    """

    visual = VisualSpec('agents')

    def __init__(
        self,
        label: str,
        *,
        x: float | Array = 0.0,
        y: float | Array = 0.0,
        z: float | Array = 0.0,
        radius: Annotated[float, RealSet.open(0, oo)] = 5.0,
        separation_radius: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        cohesion: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        separation: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        alignment: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        min_speed: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        max_speed: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.separation_radius = separation_radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.min_speed = min_speed
        self.max_speed = max_speed


class FlightInteraction(MovementInteraction):
    """Flight of every boid by its velocity `("x'", "y'")` in one step.

    The boids fly on the torus of side `width` by `height`.

    Every boid moves in its own activation, after the rules and member interactions listed before it.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of boids.

        probability:
            The probability that a boid moves in a step.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import FlightInteraction
        >>> FlightInteraction('flight', target='flock').schedule
        'member'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        probability: Annotated[float, RealSet.closed(0, 1)] = 1.0,
    ) -> None:
        super().__init__(label, target=target, step=self._fly, probability=probability, schedule='member')

    @staticmethod
    def _get_velocity(boid: AgentViewProtocol) -> Array:
        """Return the velocity of a boid as a vector."""
        return np.array([boid.get_value("x'"), boid.get_value("y'")], dtype=float)

    @staticmethod
    def _fly(agent: AgentViewProtocol, system: SystemViewProtocol) -> tuple[float, ...]:
        """Return the position of a boid after one step along its velocity."""
        boid = cast(BoidView, agent)
        position = np.asarray(boid.location, dtype=float) + FlightInteraction._get_velocity(boid)
        return tuple(float(item) for item in np.mod(position, np.array([boid.width, boid.height])))


class FlockingInteraction(ZoneCouplingInteraction):
    """Reynolds' steering of every boid by the boids it sees, with a bounded speed.

    It steers a boid toward, away from, and along those boids.

    The velocity of a boid gains the mean offset to the boids within `radius` weighted by `cohesion`, loses the summed
    offset to the boids within `separation_radius` weighted by `separation`, and gains the mean difference of their
    velocities from its own weighted by `alignment`. Its speed is then clipped between `min_speed` and `max_speed`.
    Offsets are the shortest ones on the torus of side `width` by `height`. It is the agent form of coupling over two
    radii of one priority, with a clamp of the norm.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of boids.

        phase:
            The phase of the step the steering runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import FlockingInteraction
        >>> [spec.reduce for spec in FlockingInteraction('flocking', target='flock').radii]
        ['mean', 'sum', 'mean']
    """

    def __init__(self, label: str, *, target: Role, phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            write=("x'", "y'"),
            radii=(
                RadiusSpec('radius', self._get_cohesion, reduce='mean'),
                RadiusSpec('separation_radius', self._get_separation),
                RadiusSpec('radius', self._get_alignment, reduce='mean'),
            ),
            clamp=ClampSpec(min_norm='min_speed', max_norm='max_speed'),
            phase=phase,
        )

    @staticmethod
    def _get_offset(agent: BoidView, other: BoidView) -> Array:
        """Return the shortest offset from a boid to another on the torus."""
        size = np.array([agent.width, agent.height])
        offset = np.asarray(other.location, dtype=float) - np.asarray(agent.location, dtype=float)
        return (offset + size / 2) % size - size / 2

    def _get_cohesion(self, agent: BoidView, other: BoidView, system: SystemView) -> Array:
        """Return the pull of a boid toward another."""
        return agent.cohesion * self._get_offset(agent, other)

    def _get_separation(self, agent: BoidView, other: BoidView, system: SystemView) -> Array:
        """Return the push of a boid away from a close one."""
        return -agent.separation * self._get_offset(agent, other)

    @staticmethod
    def _get_alignment(agent: BoidView, other: BoidView, system: SystemView) -> Array:
        """Return the turn of a boid toward the velocity of another."""
        return agent.alignment * (FlightInteraction._get_velocity(other) - FlightInteraction._get_velocity(agent))


class HeadingInteraction(Interaction):
    """Steering of the velocity of every boid by cohesion, separation, and alignment.

    The speed stays bounded.

    The velocity of a boid gains the mean offset to the boids within `radius` weighted by `cohesion`, loses the summed
    offset to the boids within `separation_radius` weighted by `separation`, and gains the mean difference of their
    velocities from its own weighted by `alignment`. Its speed is then clipped between `min_speed` and `max_speed`.
    Offsets are the shortest ones of the space. Every boid steers in its own activation.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of boids.

        phase:
            The phase of the step the steering runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import HeadingInteraction
        >>> HeadingInteraction('heading', target='flock').target
        'flock'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, phase: str | None = None) -> None:
        super().__init__(label)
        self.target = target
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the steering rule of every boid.

        Args:
            context:
                The namespace of the boids.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self.steer, phase=self.phase)]

    @staticmethod
    def _get_velocity(boid: SpatialBoidView) -> Array:
        """Return the velocity of a boid as a vector."""
        return np.array([boid.get_value(name) for name in ("x'", "y'", "z'")], dtype=float)

    @staticmethod
    def _get_steering(boid: SpatialBoidView, others: list[SpatialBoidView]) -> Array:
        """Return the change of the velocity of a boid from the boids it sees."""
        offsets = np.array([boid.get_displacement_to(other) for other in others])
        velocities = np.array([HeadingInteraction._get_velocity(other) for other in others])
        close = np.linalg.norm(offsets, axis=1) < boid.separation_radius
        return (
            boid.cohesion * offsets.mean(axis=0)
            - boid.separation * offsets[close].sum(axis=0)
            + boid.alignment * (velocities.mean(axis=0) - HeadingInteraction._get_velocity(boid))
        )

    def steer(self, agent: SpatialBoidView, system: SystemView) -> None:
        """Steer a boid by the boids it sees, with its speed clipped.

        Args:
            agent:
                The boid.

            system:
                The system.
        """
        del system
        if not agent.alive:
            return
        others = [cast(SpatialBoidView, other) for other in agent.get_neighbors(agent.radius)]
        velocity = self._get_velocity(agent)
        if others:
            velocity = velocity + self._get_steering(agent, others)
        speed = float(np.linalg.norm(velocity))
        velocity = velocity * np.clip(speed, agent.min_speed, agent.max_speed) / speed if speed else velocity
        for name, item in zip(("x'", "y'", "z'"), velocity, strict=True):
            agent.set_value(name, float(item))


class MorsePotentialInteraction(Interaction):
    """Attraction at long range and repulsion at short range.

    The potential is `U(r) = C_r exp(-r / l_r) - C_a exp(-r / l_a)`, and every member feels the force `-U'(r)` along
    the line of centres from each member within the cutoff, `x_i'' += sum_j -U'(r_ij) (x_i - x_j) / r_ij`. It holds a
    flock of birds, a school of fish, or a swarm of robots together at a spacing.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of members.

        cutoff:
            The distance beyond which pairs do not interact.

        C_a:
            The strength of the attraction.

        l_a:
            The range of the attraction.

        C_r:
            The strength of the repulsion.

        l_r:
            The range of the repulsion.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import MorsePotentialInteraction
        >>> MorsePotentialInteraction('flocking', target='birds', cutoff=10.0).l_a
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        cutoff: Annotated[float, RealSet.open(0, oo)] = 1.0,
        C_a: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        l_a: Annotated[float, RealSet.open(0, oo)] = 2.0,
        C_r: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        l_r: Annotated[float, RealSet.open(0, oo)] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.C_a = C_a
        self.l_a = l_a
        self.C_r = C_r
        self.l_r = l_r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the force of the members within the cutoff on every member.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, x = context, context.target.x
            separation = x - s.target.other.x
            r = sqrt(separation.dot_product(separation))
            distance = SR.var('r', domain='positive')
            energy = s.C_r * exp(-distance / s.l_r) - s.C_a * exp(-distance / s.l_a)
            slope = energy.diff(distance).subs({distance: r})
            force = neighbors(-slope * separation / r, RadiusNeighborhood(self.cutoff))
            self.laws_ = [RateLaw(x.diff(s.t, 2), force)]


class SelfPropulsionInteraction(Interaction):
    """Rayleigh self-propulsion, `x'' = (alpha - beta |x'|²) x'`, which drives points to the speed `sqrt(alpha / beta)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or group.

        alpha:
            The propulsion.

        beta:
            The friction at high speed.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SelfPropulsionInteraction
        >>> SelfPropulsionInteraction('flight', target='birds', alpha=1.5).alpha
        1.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        beta: Annotated[float, RealSet.open(0, oo)] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.beta = beta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the propulsion of the velocity of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x = s.target.x
            velocity = x.diff(t)
            self.laws_ = [RateLaw(x.diff(t, 2), (s.alpha - s.beta * velocity.dot_product(velocity)) * velocity)]


class SpatialFlightInteraction(MovementInteraction):
    """Flight of every boid by its velocity in one step.

    The boundary of its space acts on the new position.

    Every boid moves in its own activation, after the rules and member interactions listed before it.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of boids.

        probability:
            The probability that a boid moves in a step.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SpatialFlightInteraction
        >>> SpatialFlightInteraction('flight', target='flock').schedule
        'member'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        probability: Annotated[float, RealSet.closed(0, 1)] = 1.0,
    ) -> None:
        super().__init__(label, target=target, step=self._fly, probability=probability, schedule='member')

    @staticmethod
    def _fly(agent: AgentViewProtocol, system: SystemViewProtocol) -> Array:
        """Return the position of a boid after one step along its velocity."""
        boid = cast(SpatialBoidView, agent)
        return np.asarray(boid.location, dtype=float) + HeadingInteraction._get_velocity(boid)


class BoidsSystem(BaseSystem):
    """Reynolds' boids in two dimensions.

    They are birds that flock by local rules without a leader.

    The boids start at random positions with random headings on a torus. Alignment makes neighbours fly in the same
    direction and cohesion pulls them together, so the flock becomes polarized while separation keeps the birds apart.
    The system has the group `flock` and the velocities `"x'"` and `"y'"` of its boids.

    Args:
        n_boids:
            The number of boids.

        size:
            The side of the square torus.

        radius:
            The distance within which a boid sees others.

        cohesion:
            The weight of cohesion.

        separation:
            The weight of separation.

        alignment:
            The weight of alignment.

        random_state:
            The seed or generator of the initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BoidsSystem
        >>> BoidsSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'biology/collective_behaviour'
    topics = ('agents', 'flocking', 'self-organization', 'continuous space')
    references = ('Reynolds 1987', 'Downey 2018')
    until = 100
    dt = 1

    def __init__(
        self,
        n_boids: int = 40,
        *,
        size: float = 30.0,
        radius: float = 5.0,
        cohesion: float = 0.01,
        separation: float = 0.1,
        alignment: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_boids = n_boids
        self.size = size
        self.radius = radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Reynolds' boids in two dimensions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        headings = rng.uniform(0.0, 2 * math.pi, self.n_boids)
        positions = rng.uniform(0.0, self.size, (self.n_boids, 2))
        space = ContinuousSpace(extent=(self.size, self.size), boundary='periodic')
        flock = BoidElement(
            'flock',
            x=positions[:, 0],
            y=positions[:, 1],
            radius=self.radius,
            cohesion=self.cohesion,
            separation=self.separation,
            alignment=self.alignment,
            width=self.size,
            height=self.size,
            values={"x'": np.cos(headings), "y'": np.sin(headings)},
            size=self.n_boids,
            space=space,
        )
        interactions = [FlockingInteraction('flocking', target='flock'), FlightInteraction('flight', target='flock')]
        self.system_ = System(
            self.label,
            elements=[flock],
            interactions=interactions,
            space=space,
        )


class Boids3dSystem(BaseSystem):
    """Boids in three dimensions, birds that flock by local rules in a periodic cube.

    The boids start at random positions with random headings. Every step a boid steers by the boids within `radius` and
    then flies along its velocity, so with alignment alone the flock turns to one heading and keeps its spread in depth.
    The system has the group `flock` and the velocities `"x'"`, `"y'"`, and `"z'"` of its boids.

    Args:
        n_boids:
            The number of boids.

        size:
            The side of the periodic cube.

        radius:
            The distance within which a boid sees others.

        cohesion:
            The weight of cohesion.

        separation:
            The weight of separation.

        alignment:
            The weight of alignment.

        random_state:
            The seed or generator of the initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import Boids3dSystem
        >>> Boids3dSystem(10).prepare().system_.space.extent
        (20.0, 20.0, 20.0)
    """

    field = 'biology/collective_behaviour'
    topics = ('agents', 'flocking', 'self-organization', 'continuous space', 'three dimensions')
    references = ('Reynolds 1987', 'Downey 2018')
    until = 100
    dt = 1

    def __init__(
        self,
        n_boids: int = 50,
        *,
        size: float = 20.0,
        radius: float = 6.0,
        cohesion: float = 0.01,
        separation: float = 0.1,
        alignment: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_boids = n_boids
        self.size = size
        self.radius = radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Boids in three dimensions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        headings = rng.normal(size=(self.n_boids, 3))
        velocities = headings / np.linalg.norm(headings, axis=1, keepdims=True)
        positions = rng.uniform(0.0, self.size, (self.n_boids, 3))
        space = ContinuousSpace(extent=(self.size, self.size, self.size), boundary='periodic')
        flock = SpatialBoidElement(
            'flock',
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            radius=self.radius,
            cohesion=self.cohesion,
            separation=self.separation,
            alignment=self.alignment,
            values={"x'": velocities[:, 0], "y'": velocities[:, 1], "z'": velocities[:, 2]},
            size=self.n_boids,
            space=space,
        )
        interactions = [
            HeadingInteraction('heading', target='flock'),
            SpatialFlightInteraction('flight', target='flock'),
        ]
        self.system_ = System(
            self.label,
            elements=[flock],
            interactions=interactions,
            space=space,
        )


class BirdSwarmSystem(BaseSystem):
    """A swarm of self-propelled birds held together by a pair potential.

    The potential attracts far and repels near.

    Every bird is driven to the speed `sqrt(alpha / beta)` and feels the Morse-type potential of every other bird, so
    the swarm stays together and organizes into a flock or a rotating mill. The system has the positions `birds.x` and
    the velocities `birds.x'`.

    Args:
        n_birds:
            The number of birds.

        alpha:
            The propulsion.

        beta:
            The friction at high speed.

        C_a:
            The strength of the attraction.

        l_a:
            The range of the attraction.

        C_r:
            The strength of the repulsion.

        l_r:
            The range of the repulsion.

        random_state:
            The seed or generator of the initial positions and velocities.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BirdSwarmSystem
        >>> BirdSwarmSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'biology/collective_behaviour'
    topics = ('swarms', 'particles', 'active matter', 'self-propulsion')
    references = ("D'Orsogna et al. 2006", 'Gros 2015')
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        n_birds: int = 30,
        *,
        alpha: float = 1.0,
        beta: float = 0.5,
        C_a: float = 1.0,
        l_a: float = 2.0,
        C_r: float = 2.0,
        l_r: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_birds = n_birds
        self.alpha = alpha
        self.beta = beta
        self.C_a = C_a
        self.l_a = l_a
        self.C_r = C_r
        self.l_r = l_r
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the birds at random places and velocities held together by a pair potential.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        positions, velocities = rng.uniform(-3.0, 3.0, (self.n_birds, 2)), rng.normal(0.0, 1.0, (self.n_birds, 2))
        plane = ContinuousSpace(n_dim=2)
        birds = PointElement('birds', x=positions, values={"x'": velocities}, size=self.n_birds, space=plane)
        potential = MorsePotentialInteraction(
            'flocking',
            target='birds',
            cutoff=10 * self.l_a,
            C_a=self.C_a,
            l_a=self.l_a,
            C_r=self.C_r,
            l_r=self.l_r,
        )
        flight = SelfPropulsionInteraction('flight', target='birds', alpha=self.alpha, beta=self.beta)
        self.system_ = System(self.label, elements=[birds], interactions=[potential, flight], space=plane)
