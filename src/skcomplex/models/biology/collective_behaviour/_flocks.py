"""Collective behaviour models: flocks and swarms that move together."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ....core.elements import GroupElement, PointElement
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ..._catalog import register_model
from ._boid import BoidElement
from ._flight import FlightInteraction
from ._flocking import FlockingInteraction
from ._morse_potential import MorsePotentialInteraction
from ._self_propulsion import SelfPropulsionInteraction


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
        >>> boids(10).elements[0].check().prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    headings = rng.uniform(0.0, 2 * math.pi, n_boids)
    flock = GroupElement(
        'flock',
        element=BoidElement,
        size=n_boids,
        values={
            'vx': np.cos(headings),
            'vy': np.sin(headings),
            'radius': radius,
            'cohesion': cohesion,
            'separation': separation,
            'alignment': alignment,
            'width': size,
            'height': size,
        },
        location=[tuple(point) for point in rng.uniform(0.0, size, (n_boids, 2))],
    )
    interactions = [FlockingInteraction('flocking', target='flock'), FlightInteraction('flight', target='flock')]
    return System(
        elements=[flock],
        interactions=interactions,
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )


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
            The model, with the positions `birds.x` and the velocities `birds.x_d1`.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import bird_swarm
        >>> bird_swarm(5).elements[0].check().prepare().size_
        5
    """
    rng = np.random.default_rng(random_state)
    positions, velocities = rng.uniform(-3.0, 3.0, (n_birds, 2)), rng.normal(0.0, 1.0, (n_birds, 2))
    birds = GroupElement('birds', element=PointElement, size=n_birds, values={'x': positions, 'x_d1': velocities})
    potential = MorsePotentialInteraction(
        'flocking',
        first='birds',
        second='birds',
        cutoff=10 * l_a,
        C_a=C_a,
        l_a=l_a,
        C_r=C_r,
        l_r=l_r,
    )
    flight = SelfPropulsionInteraction('flight', target='birds', alpha=alpha, beta=beta)
    return System(elements=[birds], interactions=[potential, flight], space=ContinuousSpace(n_dim=2))
