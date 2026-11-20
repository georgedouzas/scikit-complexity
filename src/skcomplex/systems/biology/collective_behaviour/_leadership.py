"""Leadership in animal groups.

A few informed individuals steer a Couzin shoal.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.agents import MovementInteraction
from ._zones import ClampSpec, RadiusSpec, ZoneCouplingInteraction


class ShoalMemberView(AgentViewProtocol, Protocol):
    """View of a member of a shoal, with its states and parameters as attributes.

    The rules of `ShoalMemberElement` read them with checked types.
    """

    x: float
    y: float
    vx: float
    vy: float
    omega: float
    deviation: float
    preferred: float
    alpha: float
    rho: float
    speed: float
    max_turn: float
    noise: float
    side: float


class ShoalMemberElement(Element):
    """Fish of a Couzin shoal that avoids close neighbours and follows farther ones.

    It may know a preferred direction.

    Every step `SchoolingInteraction` turns a fish toward its desired direction by at most `max_turn`. When neighbours
    lie within the zone of repulsion `alpha`, it points away from them. Otherwise it points toward the neighbours
    within the zone of interaction `rho` and along their headings and its own. An informed fish adds its preferred
    direction with the weight `omega`, while a naive fish has `omega = 0`. It then draws a Gaussian angular noise
    `deviation` of standard deviation `noise`, and `SwimmingInteraction` turns it by the noise and moves it at the
    speed `speed`.

    Args:
        label:
            The label of the fish or the group.

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

        side:
            The side of the square torus.

        values:
            Further values by name, such as the coordinates `x` and `y` of the fish on the torus.

        size:
            The number of fish, or `None` for one.

        space:
            The torus the fish swim on, whose axes `x` and `y` place them, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ShoalMemberElement
        >>> ShoalMemberElement('shoal', values={'x': 5.0, 'y': 5.0}, size=10).size
        10
    """

    visual = VisualSpec('agents', {'color_by': 'omega'})

    def __init__(
        self,
        label: str,
        *,
        vx: object = 1.0,
        vy: object = 0.0,
        omega: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        deviation: object = 0.0,
        preferred: float = 0.0,
        alpha: Annotated[float, RealSet.open(0, oo)] = 1.0,
        rho: Annotated[float, RealSet.open(0, oo)] = 6.0,
        speed: Annotated[object, RealSet.open(0, oo)] = 0.3,
        max_turn: Annotated[float, RealSet.open(0, oo)] = 0.4,
        noise: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        side: Annotated[float, RealSet.open(0, oo)] = 100.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
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
        self.side = side


class SchoolingInteraction(ZoneCouplingInteraction):
    """Couzin's turn of every fish toward a desired direction, by at most `max_turn`.

    Zones around the fish set the desired direction.

    When neighbours lie within the zone of repulsion `alpha`, the desired direction points away from them, and
    repulsion overrides the other zones. Otherwise it points toward the neighbours within the zone of interaction `rho`
    and along their headings and the fish's own, and an informed fish adds its preferred direction with the weight
    `omega`. A fish with no neighbour keeps its heading. Offsets are the shortest ones on the torus of side `side`. It
    is the agent form of coupling over two radii of different priorities, with a clamp of the turn.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of fish.

        phase:
            The phase of the step the turn runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SchoolingInteraction
        >>> [spec.priority for spec in SchoolingInteraction('schooling', target='shoal').radii]
        [1, 0, 0]
    """

    def __init__(self, label: str, *, target: Role, phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            write=('vx', 'vy'),
            combine=self._get_desired,
            radii=(
                RadiusSpec('alpha', self._get_repulsion, priority=1),
                RadiusSpec('rho', self._get_attraction),
                RadiusSpec('rho', self._get_alignment),
            ),
            clamp=ClampSpec(max_angle='max_turn'),
            phase=phase,
        )

    @staticmethod
    def _get_unit(vector: Array) -> Array:
        """Return a vector scaled to unit length, or the vector itself when it is zero."""
        norm = float(np.hypot(*vector))
        return vector / norm if norm > 0 else vector

    def _get_direction(self, agent: ShoalMemberView, other: ShoalMemberView) -> Array:
        """Return the unit direction from a fish to another.

        It lies along the shortest offset on the torus.
        """
        side = agent.side
        offset = np.asarray(other.location, dtype=float) - np.asarray(agent.location, dtype=float)
        return self._get_unit((offset + side / 2) % side - side / 2)

    def _get_repulsion(self, agent: ShoalMemberView, other: ShoalMemberView, system: SystemView) -> Array:
        """Return the direction away from a close neighbour."""
        return -self._get_direction(agent, other)

    def _get_attraction(self, agent: ShoalMemberView, other: ShoalMemberView, system: SystemView) -> Array:
        """Return the direction toward a neighbour."""
        return self._get_direction(agent, other)

    @staticmethod
    def _get_alignment(agent: ShoalMemberView, other: ShoalMemberView, system: SystemView) -> Array:
        """Return the heading of a neighbour."""
        return np.array([other.vx, other.vy])

    def _get_desired(
        self,
        agent: AgentViewProtocol,
        system: SystemViewProtocol,
        reduction: Array | None,
        priority: int | None,
    ) -> Array:
        """Return the desired direction of a fish.

        It comes from repulsion, or from attraction, alignment, and the preferred direction.
        """
        fish = cast(ShoalMemberView, agent)
        heading = np.array([fish.vx, fish.vy])
        if reduction is None:
            return heading
        if priority == 1:
            desired = self._get_unit(reduction)
        else:
            goal = np.array([math.cos(fish.preferred), math.sin(fish.preferred)])
            desired = self._get_unit(self._get_unit(heading + reduction) + fish.omega * goal)
        return desired if float(np.hypot(*desired)) > 0 else heading


class SwimmingInteraction(MovementInteraction):
    """Swim of every fish by `speed` along its heading.

    The heading is turned first by its angular noise `deviation`.

    The heading `(vx, vy)` stays a unit vector, and the fish moves on the torus of side `side`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of fish.

        probability:
            The probability that a fish swims in a step.

        phase:
            The phase of the step the swim runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SwimmingInteraction
        >>> SwimmingInteraction('swimming', target='shoal', phase='swim').phase
        'swim'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        probability: Annotated[float, RealSet.closed(0, 1)] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            step=self._swim,
            probability=probability,
            schedule='member',
            phase=phase,
        )

    @staticmethod
    def _swim(agent: AgentViewProtocol, system: SystemViewProtocol) -> tuple[float, ...]:
        """Turn the heading of a fish by its angular noise and return its new position.

        The position is the one after one swim.
        """
        fish = cast(ShoalMemberView, agent)
        angle = math.atan2(fish.vy, fish.vx) + fish.deviation
        fish.vx, fish.vy = math.cos(angle), math.sin(angle)
        position = np.asarray(fish.location, dtype=float) + fish.speed * np.array([fish.vx, fish.vy])
        return tuple(float(item) for item in np.mod(position, fish.side))


class CouzinLeadershipSystem(BaseSystem):
    """The Couzin shoal, in which a small informed minority leads a group.

    The group does not know who is informed.

    The fish start in a disk of radius 5 at the center of a torus with random headings, and a fraction `informed` of
    them prefers the direction at the angle `preferred` with the weight `omega`. Repulsion, attraction, and alignment
    keep the shoal together, and the informed fish pull its mean heading toward their direction. The accuracy, the
    closeness of the group's heading to the preferred direction, grows with the size of the minority, and a minority of
    about 10% already steers a group of 50. The system has the group `shoal`, the unit headings `vx` and `vy` of its
    fish, and the weight `omega` of each, positive for the informed ones.

    Args:
        n_fish:
            The number of fish.

        informed:
            The fraction of informed fish.

        omega:
            The weight of the preferred direction of the informed fish.

        preferred:
            The angle of the preferred direction.

        side:
            The side of the square torus.

        random_state:
            The seed or generator of the initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import CouzinLeadershipSystem
        >>> int((CouzinLeadershipSystem(20, informed=0.2).prepare().system_.elements[0].omega > 0).sum())
        4
    """

    field = 'biology/collective_behaviour'
    topics = ('agents', 'schooling', 'leadership', 'collective decision', 'continuous space')
    references = ('Couzin et al. 2005', 'Mihailović et al. 2023')
    until = 300
    dt = 1

    def __init__(
        self,
        n_fish: int = 50,
        *,
        informed: float = 0.1,
        omega: float = 0.5,
        preferred: float = 0.0,
        side: float = 100.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_fish = n_fish
        self.informed = informed
        self.omega = omega
        self.preferred = preferred
        self.side = side
        self.random_state = random_state

    @staticmethod
    def _draw_deviation(agent: ShoalMemberView, system: SystemView) -> float:
        """Return a Gaussian angular noise of the turn of a fish.

        The noise has the standard deviation `noise`.
        """
        return float(system.rng.normal(0.0, agent.noise))

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Couzin shoal.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        radius = 5.0 * np.sqrt(rng.uniform(0.0, 1.0, self.n_fish))
        place = rng.uniform(0.0, 2 * math.pi, self.n_fish)
        headings = rng.uniform(0.0, 2 * math.pi, self.n_fish)
        weights = np.zeros(self.n_fish)
        weights[: round(self.informed * self.n_fish)] = self.omega
        locations = self.side / 2 + np.column_stack([radius * np.cos(place), radius * np.sin(place)])
        shoal = ShoalMemberElement(
            'shoal',
            vx=np.cos(headings),
            vy=np.sin(headings),
            omega=weights,
            preferred=self.preferred,
            side=self.side,
            values={'x': locations[:, 0], 'y': locations[:, 1]},
            size=self.n_fish,
            space=ContinuousSpace(extent=(self.side, self.side), boundary='periodic'),
        )
        self.system_ = System(
            self.label,
            elements=[shoal],
            interactions=[
                SchoolingInteraction('schooling', target='shoal', phase='turn'),
                Interaction(
                    'deviation',
                    roles={'target': 'shoal'},
                    laws=lambda s: [
                        UpdateLaw(
                            s.target,
                            lambda agent, system: setattr(
                                agent,
                                'deviation',
                                float(self._draw_deviation(agent, system)),
                            ),
                            phase='turn',
                        ),
                    ],
                ),
                SwimmingInteraction('swimming', target='shoal', phase='swim'),
            ],
            solver=MesaSolver(phases=[PhaseSpec('turn', activation='simultaneous'), PhaseSpec('swim')]),
        )
