"""Collective behaviour models: flocks and swarms that move together."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement, PointElement
from ....core.interactions import PairCouplingInteraction, RateInteraction
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._boid_view import BoidView
from ._flight import FlightInteraction
from ._flocking import FlockingInteraction


class BoidElement(Element):
    """Boid of Reynolds, a bird that steers by three rules toward, away from, and along its neighbours.

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
        >>> from skcomplex.models.biology.collective_behaviour import BoidElement
        >>> GroupElement('flock', element=BoidElement, size=10, location=(5.0, 5.0)).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'vx'}),
        *,
        vx: Annotated[float, StateSpec()] = 1.0,
        vy: Annotated[float, StateSpec()] = 0.0,
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


class SelfPropulsionInteraction(RateInteraction):
    """Rayleigh self-propulsion, `v' = (alpha - beta |v|²) v`, which drives points to the speed `sqrt(alpha / beta)`.

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
        >>> from skcomplex.models.biology.collective_behaviour import SelfPropulsionInteraction
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


class MorsePotentialInteraction(PairCouplingInteraction):
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
        >>> from skcomplex.models.biology.collective_behaviour import MorsePotentialInteraction
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
