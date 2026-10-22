"""Collective behaviour models: flocks and swarms that move together."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar, cast

import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, VisualSpec, parameter, role, state
from ....core.model import (
    AgentType,
    ContinuousSpace,
    Interaction,
    Law,
    PairPotential,
    Particle,
    Rate,
    Rule,
    Symbols,
    System,
)
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView


def _wrapped(offset: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Return the shortest offset between two points of a torus."""
    return (offset + size / 2) % size - size / 2


def _limited(velocity: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return a velocity whose speed is clipped between two bounds."""
    speed = float(np.hypot(*velocity))
    if speed == 0.0:
        return velocity
    return velocity * min(max(speed, low), high) / speed


class Boid(AgentType):
    """Boid of Reynolds, a bird that steers by three rules toward, away from, and along its neighbours.

    Every step a boid sees the boids within `radius`. Cohesion steers it toward their center, separation away from
    those closer than `separation_radius`, and alignment toward their mean velocity. Its speed stays between
    `min_speed` and `max_speed`, and it moves on a torus of side `width` by `height`.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial position, or one position per boid.

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
        >>> from skcomplex.models.biology.collective_behaviour import Boid
        >>> Boid.group('flock', 10, location=(5.0, 5.0)).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'vx'})
    vx: float = state(default=1.0)
    vy: float = state(default=0.0)
    radius: float = parameter(default=5.0, domain=Domain.POSITIVE)
    separation_radius: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    cohesion: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    separation: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    alignment: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    min_speed: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    max_speed: float = parameter(default=1.0, domain=Domain.POSITIVE)
    width: float = parameter(default=30.0, domain=Domain.POSITIVE)
    height: float = parameter(default=30.0, domain=Domain.POSITIVE)

    def _steering(self, agent: AgentView, neighbours: list[AgentView], size: np.ndarray) -> np.ndarray:
        """Return the change of velocity from cohesion, separation, and alignment."""
        here = np.asarray(agent.location, dtype=float)
        offsets = np.array([_wrapped(np.asarray(other.location) - here, size) for other in neighbours])
        velocities = np.array([(other.vx, other.vy) for other in neighbours])
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        close = offsets[distances < agent.separation_radius]
        return (
            agent.cohesion * offsets.mean(axis=0)
            - agent.separation * close.sum(axis=0)
            + agent.alignment * (velocities.mean(axis=0) - (agent.vx, agent.vy))
        )

    def laws(self, s: Symbols) -> list[Law]:
        """Return the flocking rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.flock)]

    def flock(self, agent: AgentView, model: ModelView) -> None:
        """Steer by the three rules, limit the speed, and move.

        Args:
            agent:
                The boid.

            model:
                The model.
        """
        size = np.array([agent.width, agent.height])
        velocity = np.array([agent.vx, agent.vy])
        neighbours = agent.neighbors(radius=agent.radius)
        if neighbours:
            velocity = velocity + self._steering(agent, neighbours, size)
        velocity = _limited(velocity, agent.min_speed, agent.max_speed)
        agent.vx, agent.vy = float(velocity[0]), float(velocity[1])
        agent.move_to(tuple(np.mod(np.asarray(cast('tuple[float, float]', agent.location)) + velocity, size)))


class SelfPropulsion(Interaction):
    """Rayleigh self-propulsion, `v' = (alpha - beta |v|²) v / m`, which drives particles to speed `sqrt(alpha / beta)`.

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
        >>> from skcomplex.models.biology.collective_behaviour import SelfPropulsion
        >>> SelfPropulsion('flight', target='birds', alpha=1.5).alpha
        1.5
    """

    target: str = role()
    alpha: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.5, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the velocity change `(alpha - beta |v|²) v / m`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        velocity = s.target.v
        return [Rate(velocity, (s.alpha - s.beta * velocity.dot_product(velocity)) * velocity / s.target.m)]


class MorsePotential(PairPotential):
    """Attraction at long range and repulsion at short range, `C_r exp(-r / l_r) - C_a exp(-r / l_a)`.

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

        C_a:
            The strength of the attraction.

        l_a:
            The range of the attraction.

        C_r:
            The strength of the repulsion.

        l_r:
            The range of the repulsion.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import MorsePotential
        >>> MorsePotential('flocking', first='birds', second='birds', cutoff=10.0).l_a
        2.0
    """

    C_a: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    l_a: float = parameter(default=2.0, domain=Domain.POSITIVE)
    C_r: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    l_r: float = parameter(default=0.5, domain=Domain.POSITIVE)

    def pair_energy(self, s: Symbols, r: Expression) -> Expression:
        """Return `C_r exp(-r / l_r) - C_a exp(-r / l_a)`.

        Args:
            s:
                The namespace of the symbols the energy can use.

            r:
                The distance between the two particles.

        Returns:
            energy:
                The pair energy as a symbolic expression.
        """
        return s.C_r * exp(-r / s.l_r) - s.C_a * exp(-r / s.l_a)


@register_model(
    field='biology/collective_behaviour',
    tags=('agents', 'flocking', 'self-organization', 'continuous space'),
    references=('Reynolds 1987', 'Downey 2018'),
    defaults={'until': 100, 'dt': 1},
)
def boids(
    n_boids: int = 40,
    *,
    size: float = 30.0,
    radius: float = 5.0,
    cohesion: float = 0.01,
    separation: float = 0.1,
    alignment: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Reynolds' boids in two dimensions, birds that flock by local rules without a leader.

    The boids start at random positions with random headings on a torus. Alignment makes neighbours fly in the same
    direction and cohesion pulls them together, so the flock becomes polarized while separation keeps the birds apart.

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

    Returns:
        system:
            The model, with the group `flock` and the velocities `vx` and `vy` of its boids.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import boids
        >>> boids(10).elements[0].size
        10
    """
    rng = np.random.default_rng(random_state)
    headings = rng.uniform(0.0, 2 * math.pi, n_boids)
    flock = Boid.group(
        'flock',
        n_boids,
        location=[tuple(point) for point in rng.uniform(0.0, size, (n_boids, 2))],
        vx=np.cos(headings),
        vy=np.sin(headings),
        radius=radius,
        cohesion=cohesion,
        separation=separation,
        alignment=alignment,
        width=size,
        height=size,
    )
    return System(components=[flock], space=ContinuousSpace(size, size, boundary='periodic'))


@register_model(
    field='biology/collective_behaviour',
    tags=('swarms', 'particles', 'active matter', 'self-propulsion'),
    references=("D'Orsogna et al. 2006", 'Gros 2015'),
    defaults={'until': 50.0, 'dt': 0.5},
)
def bird_swarm(
    n_birds: int = 30,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    C_a: float = 1.0,
    l_a: float = 2.0,
    C_r: float = 2.0,
    l_r: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a swarm of self-propelled birds held together by a pair potential that attracts far and repels near.

    Every bird is driven to the speed `sqrt(alpha / beta)` and feels the Morse-type potential of every other bird, so
    the swarm stays together and organizes into a flock or a rotating mill.

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

    Returns:
        system:
            The model, with the positions `birds.x` and the velocities `birds.v`.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import bird_swarm
        >>> bird_swarm(5).elements[0].size
        5
    """
    rng = np.random.default_rng(random_state)
    positions, velocities = rng.uniform(-3.0, 3.0, (n_birds, 2)), rng.normal(0.0, 1.0, (n_birds, 2))
    birds = Particle.group('birds', n_birds, x=positions, v=velocities)
    potential = MorsePotential(
        'flocking',
        first='birds',
        second='birds',
        cutoff=10 * l_a,
        C_a=C_a,
        l_a=l_a,
        C_r=C_r,
        l_r=l_r,
    )
    flight = SelfPropulsion('flight', target='birds', alpha=alpha, beta=beta)
    return System(components=[birds], interactions=[potential, flight], space=ContinuousSpace(n_dim=2))
