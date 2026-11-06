"""Leadership in animal groups: a Couzin shoal in which a few informed individuals steer the group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import ContinuousSpace, DiscreteTime
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._schooling import SchoolingInteraction
from ._shoal_member_view import ShoalMemberView
from ._swimming import SwimmingInteraction


def _draw_deviation(agent: ShoalMemberView, system: SystemView) -> float:
    """Return a Gaussian angular noise of the turn of a fish, of standard deviation `noise`."""
    return float(system.rng.normal(0.0, agent.noise))


class ShoalMemberElement(Element):
    """Fish of a Couzin shoal that avoids close neighbours, follows farther ones, and may know a preferred direction.

    Every step `SchoolingInteraction` turns a fish toward its desired direction by at most `max_turn`. When neighbours
    lie within the zone of repulsion `alpha`, it points away from them. Otherwise it points toward the neighbours
    within the zone of interaction `rho` and along their headings and its own. An informed fish adds its preferred
    direction with the weight `omega`, while a naive fish has `omega = 0`. It then draws a Gaussian angular noise
    `deviation` of standard deviation `noise`, and `SwimmingInteraction` turns it by the noise and moves it at the
    speed `speed`.

    Args:
        label:
            The label of the fish or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        vx:
            The horizontal component of the unit heading.

        vy:
            The vertical component of the unit heading.

        omega:
            The weight of the preferred direction, 0 for a naive fish.

        deviation:
            The angular noise of the last turn.

        preferred:
            The angle of the preferred direction of the informed fish.

        alpha:
            The radius of the zone of repulsion.

        rho:
            The radius of the zone of interaction.

        speed:
            The distance a fish swims per step.

        max_turn:
            The largest turning angle per step.

        noise:
            The standard deviation of the angular noise.

        size:
            The side of the square torus.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ShoalMemberElement
        >>> GroupElement('shoal', element=ShoalMemberElement, size=10, location=(5.0, 5.0)).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'omega'}),
        *,
        vx: Annotated[float, StateSpec(role='velocity')] = 1.0,
        vy: Annotated[float, StateSpec(role='velocity')] = 0.0,
        omega: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        deviation: Annotated[float, StateSpec()] = 0.0,
        preferred: Annotated[float, ParameterSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        max_turn: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        size: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.deviation = deviation
        self.preferred = preferred
        self.alpha = alpha
        self.rho = rho
        self.speed = speed
        self.max_turn = max_turn
        self.noise = noise
        self.size = size

    @classmethod
    def _get_view(cls) -> type[ShoalMemberView]:
        """Return the view of a member of a shoal, `ShoalMemberView`."""
        return ShoalMemberView


@register_system(
    field='biology/collective_behaviour',
    tags=('agents', 'schooling', 'leadership', 'collective decision', 'continuous space'),
    references=('Couzin et al. 2005', 'Mihailović et al. 2023'),
    defaults={'until': 300, 'dt': 1},
)
def couzin_leadership(
    n_fish: int = 50,
    *,
    informed: float = 0.1,
    omega: float = 0.5,
    preferred: float = 0.0,
    size: float = 100.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Couzin shoal, in which a small informed minority leads a group that does not know who is informed.

    The fish start in a disk of radius 5 at the center of a torus with random headings, and a fraction `informed` of
    them prefers the direction at the angle `preferred` with the weight `omega`. Repulsion, attraction, and alignment
    keep the shoal together, and the informed fish pull its mean heading toward their direction. The accuracy, the
    closeness of the group's heading to the preferred direction, grows with the size of the minority, and a minority
    of about 10% already steers a group of 50.

    Args:
        n_fish:
            The number of fish.

        informed:
            The fraction of informed fish.

        omega:
            The weight of the preferred direction of the informed fish.

        preferred:
            The angle of the preferred direction.

        size:
            The side of the square torus.

        random_state:
            The seed or generator of the initial positions and headings.

    Returns:
        system:
            The system, with the group `shoal`, the unit headings `vx` and `vy` of its fish, and the weight `omega` of
            each, positive for the informed ones.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import couzin_leadership
        >>> int((couzin_leadership(20, informed=0.2).elements[0].values['omega'] > 0).sum())
        4
    """
    rng = np.random.default_rng(random_state)
    radius = 5.0 * np.sqrt(rng.uniform(0.0, 1.0, n_fish))
    place = rng.uniform(0.0, 2 * math.pi, n_fish)
    headings = rng.uniform(0.0, 2 * math.pi, n_fish)
    weights = np.zeros(n_fish)
    weights[: round(informed * n_fish)] = omega
    locations = size / 2 + np.column_stack([radius * np.cos(place), radius * np.sin(place)])
    shoal = GroupElement(
        'shoal',
        element=ShoalMemberElement,
        size=n_fish,
        values={'vx': np.cos(headings), 'vy': np.sin(headings), 'omega': weights, 'preferred': preferred, 'size': size},
        location=[tuple(point) for point in locations],
    )
    return System(
        elements=[shoal],
        interactions=[
            SchoolingInteraction('schooling', target='shoal', phase='turn'),
            MapInteraction(
                'deviation',
                target='shoal',
                state='deviation',
                function=_draw_deviation,
                form='agent',
                schedule='member',
                phase='turn',
            ),
            SwimmingInteraction('swimming', target='shoal', phase='swim'),
        ],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('turn', activation='simultaneous'), PhaseSpec('swim')]),
        ),
    )
