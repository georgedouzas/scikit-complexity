"""Leadership in animal groups: a Couzin shoal in which a few informed individuals steer the group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import ContinuousSpace, DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._shoal_member_view import ShoalMemberView


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return a vector scaled to unit length, or the vector itself when it is zero."""
    norm = float(np.hypot(*vector))
    return vector / norm if norm > 0 else vector


def _turned(heading: np.ndarray, desired: np.ndarray, max_turn: float) -> np.ndarray:
    """Return the heading turned toward a desired direction by at most an angle."""
    angle = math.atan2(desired[1], desired[0]) - math.atan2(heading[1], heading[0])
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    angle = max(-max_turn, min(max_turn, angle))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([cosine * heading[0] - sine * heading[1], sine * heading[0] + cosine * heading[1]])


class ShoalMemberElement(Element):
    """Fish of a Couzin shoal that avoids close neighbours, follows farther ones, and may know a preferred direction.

    Every step a fish first sets its desired direction. When neighbours lie within the zone of repulsion `alpha`, it
    points away from them. Otherwise it points toward the neighbours within the zone of interaction `rho` and along
    their headings and its own. An informed fish adds its preferred direction with the weight `omega`, while a naive
    fish has `omega = 0`. It then turns toward the desired direction by at most `max_turn`, with Gaussian angular
    noise of standard deviation `noise`, and swims at the speed `speed`.

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
        >>> from skcomplex.models.biology.collective_behaviour import ShoalMemberElement
        >>> GroupElement('shoal', element=ShoalMemberElement, size=10, location=(5.0, 5.0)).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'omega'}),
        *,
        vx: Annotated[float, StateSpec()] = 1.0,
        vy: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
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

    def _desired(self, agent: ShoalMemberView) -> np.ndarray:
        """Return the desired direction from repulsion, or from attraction and alignment."""
        heading = np.array([agent.vx, agent.vy])
        neighbours = [other for other in agent.neighbors(radius=agent.rho) if isinstance(other, ShoalMemberView)]
        if not neighbours:
            return heading
        size = agent.size
        here = np.asarray(agent.location, dtype=float)
        offsets = np.array([other.location for other in neighbours], dtype=float) - here
        offsets = (offsets + size / 2) % size - size / 2
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        directions = offsets / distances[:, None]
        close = distances < agent.alpha
        if close.any():
            return _unit(-directions[close].sum(axis=0))
        headings = np.array([(other.vx, other.vy) for other in neighbours])
        social = heading + directions.sum(axis=0) + headings.sum(axis=0)
        goal = np.array([math.cos(agent.preferred), math.sin(agent.preferred)])
        return _unit(_unit(social) + agent.omega * goal)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the simultaneous turning and the swimming of the fish.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.turn, phase='turn'), RuleLaw(self.swim, phase='swim')]

    def turn(self, agent: ShoalMemberView, model: ModelView) -> None:
        """Turn toward the desired direction, with angular noise.

        Args:
            agent:
                The fish.

            model:
                The model.
        """
        heading = np.array([agent.vx, agent.vy])
        desired = self._desired(agent)
        if float(np.hypot(*desired)) > 0:
            heading = _turned(heading, desired, agent.max_turn)
        angle = math.atan2(heading[1], heading[0]) + float(model.rng.normal(0.0, agent.noise))
        agent.vx, agent.vy = math.cos(angle), math.sin(angle)

    def swim(self, agent: ShoalMemberView, model: ModelView) -> None:
        """Swim one step along the heading.

        Args:
            agent:
                The fish.

            model:
                The model.
        """
        position = np.asarray(cast('tuple[float, float]', agent.location)) + agent.speed * np.array(
            [agent.vx, agent.vy],
        )
        agent.move_to(tuple(np.mod(position, agent.size)))


@register_model(
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
            The model, with the group `shoal`, the unit headings `vx` and `vy` of its fish, and the weight `omega` of
            each, positive for the informed ones.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import couzin_leadership
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
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('turn', activation='simultaneous'), PhaseSpec('swim')]),
        ),
    )
