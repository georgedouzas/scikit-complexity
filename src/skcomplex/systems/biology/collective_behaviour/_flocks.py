"""Flocks and swarms that move together."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction, MapInteraction, PairInteraction, RateInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ClampSpec, ParameterSpec, RadiusSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.agents import MovementInteraction
from ...general.elements import PointElement


class BoidView(AgentView):
    """View of a boid, with its states and parameters as attributes.

    `BoidElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    vx: float
    vy: float
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
    `height`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

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

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.collective_behaviour import BoidElement
        >>> GroupElement('flock', element=BoidElement, size=10, location=(5.0, 5.0)).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'vx'}),
        *,
        vx: Annotated[float, StateSpec(role='velocity')] = 1.0,
        vy: Annotated[float, StateSpec(role='velocity')] = 0.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        separation_radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        cohesion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        separation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        alignment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        min_speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        max_speed: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        height: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.separation_radius = separation_radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.width = width
        self.height = height

    @classmethod
    def _get_view(cls) -> type[BoidView]:
        """Return the view of a boid, `BoidView`."""
        return BoidView


class SpatialBoidView(AgentView):
    """View of a boid in three dimensions.

    The view has the states and parameters of the boid as attributes.

    `SpatialBoidElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    velocity: Array
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
    with its speed between `min_speed` and `max_speed`, and `SpatialFlightInteraction` moves it along its velocity.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        velocity:
            The initial velocity, one component per dimension of the space.

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

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.collective_behaviour import SpatialBoidElement
        >>> GroupElement('flock', element=SpatialBoidElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents'),
        *,
        velocity: Annotated[npt.ArrayLike, StateSpec(vector=True, role='velocity')] = 0.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        separation_radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        cohesion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        separation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        alignment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        min_speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        max_speed: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.velocity = velocity
        self.radius = radius
        self.separation_radius = separation_radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.min_speed = min_speed
        self.max_speed = max_speed

    @classmethod
    def _get_view(cls) -> type[SpatialBoidView]:
        """Return the view of a boid, `SpatialBoidView`."""
        return SpatialBoidView


class FlightInteraction(MovementInteraction):
    """Flight of every boid by its velocity `(vx, vy)` in one step.

    The boids fly on the torus of side `width` by `height`.

    Every boid moves in its own activation, after the rules and member interactions listed before it.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, step=self._fly, probability=probability, schedule='member')

    @staticmethod
    def _fly(agent: AgentViewProtocol, system: SystemViewProtocol) -> tuple[float, ...]:
        """Return the position of a boid after one step along its velocity."""
        boid = cast(BoidView, agent)
        position = np.asarray(boid.location, dtype=float) + np.array([boid.vx, boid.vy])
        return tuple(float(item) for item in np.mod(position, np.array([boid.width, boid.height])))


class FlockingInteraction(CouplingInteraction):
    """Reynolds' steering of every boid by the boids it sees, with a bounded speed.

    It steers a boid toward, away from, and along those boids.

    The velocity of a boid gains the mean offset to the boids within `radius` weighted by `cohesion`, loses the summed
    offset to the boids within `separation_radius` weighted by `separation`, and gains the mean difference of their
    velocities from its own weighted by `alignment`. Its speed is then clipped between `min_speed` and `max_speed`.
    Offsets are the shortest ones on the torus of side `width` by `height`. It is the agent form of coupling over two
    radii of one priority, with a clamp of the norm.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of boids.

        phase:
            The phase of the step the steering runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import FlockingInteraction
        >>> [spec.reduce for spec in FlockingInteraction('flocking', target='flock').radii]
        ['mean', 'sum', 'mean']
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            write=('vx', 'vy'),
            form='agent',
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
        return agent.alignment * np.array([other.vx - agent.vx, other.vy - agent.vy])


class HeadingInteraction(MapInteraction):
    """Steering of the velocity of every boid by cohesion, separation, and alignment.

    The speed stays bounded.

    The velocity of a boid gains the mean offset to the boids within `radius` weighted by `cohesion`, loses the summed
    offset to the boids within `separation_radius` weighted by `separation`, and gains the mean difference of their
    velocities from its own weighted by `alignment`. Its speed is then clipped between `min_speed` and `max_speed`.
    Offsets are the shortest ones of the space. It is the agent form of a map of the velocity under the `'member'`
    schedule.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of boids.

        phase:
            The phase of the step the steering runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import HeadingInteraction
        >>> HeadingInteraction('heading', target='flock').schedule
        'member'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            state='velocity',
            function=self._steer,
            schedule='member',
            form='agent',
            phase=phase,
        )

    @staticmethod
    def _get_steering(boid: SpatialBoidView, others: list[SpatialBoidView]) -> Array:
        """Return the change of the velocity of a boid from the boids it sees."""
        offsets = np.array([boid.displacement_to(other) for other in others])
        velocities = np.array([other.velocity for other in others])
        close = np.linalg.norm(offsets, axis=1) < boid.separation_radius
        return (
            boid.cohesion * offsets.mean(axis=0)
            - boid.separation * offsets[close].sum(axis=0)
            + boid.alignment * (velocities.mean(axis=0) - boid.velocity)
        )

    @staticmethod
    def _steer(agent: AgentViewProtocol, system: SystemViewProtocol) -> Array:
        """Return the velocity of a boid after one step of steering.

        Its speed is clipped.
        """
        boid = cast(SpatialBoidView, agent)
        others = [cast(SpatialBoidView, other) for other in boid.neighbors(boid.radius)]
        velocity = np.asarray(boid.velocity, dtype=float)
        if others:
            velocity = velocity + HeadingInteraction._get_steering(boid, others)
        speed = float(np.linalg.norm(velocity))
        return velocity * np.clip(speed, boid.min_speed, boid.max_speed) / speed if speed else velocity


class MorsePotentialInteraction(PairInteraction):
    """Attraction at long range and repulsion at short range.

    The potential is `C_r exp(-r / l_r) - C_a exp(-r / l_a)`.

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
            The name of the position state of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

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
        >>> MorsePotentialInteraction('flocking', first='birds', second='birds', cutoff=10.0).l_a
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
        C_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l_a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        C_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        l_r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
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
        self.C_a = C_a
        self.l_a = l_a
        self.C_r = C_r
        self.l_r = l_r

    @staticmethod
    def _get_pair_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return `C_r exp(-r / l_r) - C_a exp(-r / l_a)`."""
        return s.C_r * exp(-r / s.l_r) - s.C_a * exp(-r / s.l_a)


class SelfPropulsionInteraction(RateInteraction):
    """Rayleigh self-propulsion, `v' = (alpha - beta |v|²) v`.

    It drives points to the speed `sqrt(alpha / beta)`.

    It is a rate on the position `x`, a state of order 2, whose velocity `v` is the first derivative `x_d1`.

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

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, target=target, state='x')
        self.alpha = alpha
        self.beta = beta

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the velocity change `(alpha - beta |v|²) v`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The change of the velocity.
        """
        velocity = namespace.target.x_d1
        return (namespace.alpha - namespace.beta * velocity.dot_product(velocity)) * velocity


class SpatialFlightInteraction(MovementInteraction):
    """Flight of every boid by its velocity in one step.

    The boundary of its space acts on the new position.

    Every boid moves in its own activation, after the rules and member interactions listed before it.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, step=self._fly, probability=probability, schedule='member')

    @staticmethod
    def _fly(agent: AgentViewProtocol, system: SystemViewProtocol) -> Array:
        """Return the position of a boid after one step along its velocity."""
        boid = cast(SpatialBoidView, agent)
        return np.asarray(boid.location, dtype=float) + np.asarray(boid.velocity, dtype=float)


class BoidsSystem(BaseSystem):
    """Reynolds' boids in two dimensions.

    They are birds that flock by local rules without a leader.

    The boids start at random positions with random headings on a torus. Alignment makes neighbours fly in the same
    direction and cohesion pulls them together, so the flock becomes polarized while separation keeps the birds apart.
    The system has the group `flock` and the velocities `vx` and `vy` of its boids.

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

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BoidsSystem
        >>> BoidsSystem(10).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.n_boids = n_boids
        self.size = size
        self.radius = radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of Reynolds' boids in two dimensions.

        The boids are birds that flock by local rules without a leader.
        """
        rng = np.random.default_rng(self.random_state)
        headings = rng.uniform(0.0, 2 * math.pi, self.n_boids)
        flock = GroupElement(
            'flock',
            element=BoidElement,
            size=self.n_boids,
            values={
                'vx': np.cos(headings),
                'vy': np.sin(headings),
                'radius': self.radius,
                'cohesion': self.cohesion,
                'separation': self.separation,
                'alignment': self.alignment,
                'width': self.size,
                'height': self.size,
            },
            location=[tuple(point) for point in rng.uniform(0.0, self.size, (self.n_boids, 2))],
        )
        interactions = [FlockingInteraction('flocking', target='flock'), FlightInteraction('flight', target='flock')]
        return CompositeSystem(
            self.label,
            elements=[flock],
            interactions=interactions,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )


class Boids3dSystem(BaseSystem):
    """Boids in three dimensions, birds that flock by local rules in a periodic cube.

    The boids start at random positions with random headings. Every step a boid steers by the boids within `radius` and
    then flies along its velocity, so with alignment alone the flock turns to one heading and keeps its spread in depth.
    The system has the group `flock` and the vector state `velocity` of its boids.

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
    ) -> None:
        super().__init__(label)
        self.n_boids = n_boids
        self.size = size
        self.radius = radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of Boids in three dimensions.

        The boids are birds that flock by local rules in a periodic cube.
        """
        rng = np.random.default_rng(self.random_state)
        headings = rng.normal(size=(self.n_boids, 3))
        flock = GroupElement(
            'flock',
            element=SpatialBoidElement,
            size=self.n_boids,
            values={
                'velocity': headings / np.linalg.norm(headings, axis=1, keepdims=True),
                'radius': self.radius,
                'cohesion': self.cohesion,
                'separation': self.separation,
                'alignment': self.alignment,
            },
            location=[tuple(point) for point in rng.uniform(0.0, self.size, (self.n_boids, 3))],
        )
        interactions = [
            HeadingInteraction('heading', target='flock'),
            SpatialFlightInteraction('flight', target='flock'),
        ]
        return CompositeSystem(
            self.label,
            elements=[flock],
            interactions=interactions,
            space=ContinuousSpace(extent=(self.size, self.size, self.size), boundary='periodic'),
        )


class BirdSwarmSystem(BaseSystem):
    """A swarm of self-propelled birds held together by a pair potential.

    The potential attracts far and repels near.

    Every bird is driven to the speed `sqrt(alpha / beta)` and feels the Morse-type potential of every other bird, so
    the swarm stays together and organizes into a flock or a rotating mill. The system has the positions `birds.x` and
    the velocities `birds.x_d1`.

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

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BirdSwarmSystem
        >>> BirdSwarmSystem(5).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.n_birds = n_birds
        self.alpha = alpha
        self.beta = beta
        self.C_a = C_a
        self.l_a = l_a
        self.C_r = C_r
        self.l_r = l_r
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a swarm of self-propelled birds.

        A pair potential holds the birds together.
        """
        rng = np.random.default_rng(self.random_state)
        positions, velocities = rng.uniform(-3.0, 3.0, (self.n_birds, 2)), rng.normal(0.0, 1.0, (self.n_birds, 2))
        birds = GroupElement(
            'birds',
            element=PointElement,
            size=self.n_birds,
            values={'x': positions, 'x_d1': velocities},
        )
        potential = MorsePotentialInteraction(
            'flocking',
            first='birds',
            second='birds',
            cutoff=10 * self.l_a,
            C_a=self.C_a,
            l_a=self.l_a,
            C_r=self.C_r,
            l_r=self.l_r,
        )
        flight = SelfPropulsionInteraction('flight', target='birds', alpha=self.alpha, beta=self.beta)
        return CompositeSystem(
            self.label,
            elements=[birds],
            interactions=[potential, flight],
            space=ContinuousSpace(n_dim=2),
        )
