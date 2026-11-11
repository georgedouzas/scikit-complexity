"""Active walkers that form trails by changing the ground they walk on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import MapInteraction, RateInteraction, TransferInteraction
from ....core.components.laws import Source
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.agents import BirthDeathInteraction, MovementInteraction

TRAIL = 'land/ground.V'


class GroundElement(Element):
    """Ground whose structure `G` is raised by footprints and recovers.

    A trail potential `V` smooths the structure.

    The structure relaxes back to bare ground as `G' = -G / T`, and walkers add footprints. The trail potential
    follows `V' = D ΔV + (G - V) / tau`, so it settles to the structure smoothed over a length `sqrt(D tau)`, the
    distance from which walkers notice a trail. The element holds the states and their parameters, and the system adds
    the fading, the diffusion, and the relaxation as interactions.

    Args:
        label:
            The label of the ground or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        G:
            The initial ground structure.

        V:
            The initial trail potential.

        T:
            The durability of trails, the time over which they fade.

        D:
            The diffusion of the trail potential.

        tau:
            The relaxation time of the trail potential.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import GroundElement
        >>> GroundElement('ground').T
        200.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.G = G
        self.V = V
        self.T = T
        self.D = D
        self.tau = tau


class WalkerView(AgentView):
    """View of a walker, with its states and parameters as attributes.

    `WalkerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    goal_x: float
    goal_y: float
    arrived: float
    directedness: float
    attraction: float
    intensity: float
    G_max: float
    width: float
    height: float


class WalkerElement(Element):
    """Pedestrian who walks toward a goal, drawn to existing trails.

    The walker leaves footprints that build the trails.

    The walker heads along the unit vector toward its goal plus `attraction` times the gradient of the trail potential
    `V`, after Helbing, Keltsch, and Molnár. Every step it moves to a neighbouring cell with a probability that grows
    with `exp(directedness * cos angle)`, where the angle is between the step and the heading, and adds the footprint
    `intensity (1 - G / G_max)` to the ground. The step after it arrives at its goal it leaves the green, and a new
    walker enters there with another entry point as goal, drawn at random, a renewal in place. The system adds the
    step, the footprint, the renewal, and the arrival as member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        goal_x:
            The column of the goal.

        goal_y:
            The row of the goal.

        arrived:
            Whether the walker stood at its goal at the end of the previous step, 1 or 0.

        directedness:
            How strongly the walker heads for its goal.

        attraction:
            How strongly trails draw the walker.

        intensity:
            The footprint on bare ground.

        G_max:
            The largest ground structure.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.collective_behaviour import WalkerElement
        >>> GroupElement('walkers', element=WalkerElement, size=4, location=(0, 0)).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'goal_x'}),
        *,
        goal_x: Annotated[float, StateSpec()] = 0.0,
        goal_y: Annotated[float, StateSpec()] = 0.0,
        arrived: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        directedness: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        attraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        G_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        height: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.arrived = arrived
        self.directedness = directedness
        self.attraction = attraction
        self.intensity = intensity
        self.G_max = G_max
        self.width = width
        self.height = height

    @classmethod
    def _get_view(cls) -> type[WalkerView]:
        """Return the view of a walker, `WalkerView`."""
        return WalkerView


class ActiveWalkersSystem(BaseSystem):
    """Active walkers who wear a system of trails into a green.

    They walk between the corners of the green.

    Walkers prefer existing trails to the direct way, and trails fade unless they are used, so a few shared trails
    emerge that bundle the ways between the corners, a compromise between short and comfortable routes. The system has
    the walkers `people/walkers`, the ground structure `land/ground.G`, and the trail potential `land/ground.V`.

    Args:
        size:
            The number of rows and columns of the green.

        n_walkers:
            The number of walkers.

        attraction:
            How strongly trails draw the walkers.

        T:
            The durability of trails.

        random_state:
            The seed or generator of the starting corners and goals.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ActiveWalkersSystem
        >>> [child.label for child in ActiveWalkersSystem().prepare().system_.systems]
        ['people', 'land']
    """

    field = 'biology/collective_behaviour'
    topics = ('agents', 'field', 'trail formation', 'co-simulation')
    references = ('Helbing, Keltsch, and Molnár 1997', 'arXiv:cond-mat/0012229')
    until = 300
    dt = 1

    def __init__(
        self,
        size: int = 30,
        *,
        n_walkers: int = 20,
        attraction: float = 1.0,
        T: float = 200.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.n_walkers = n_walkers
        self.attraction = attraction
        self.T = T
        self.random_state = random_state

    @staticmethod
    def _get_corners(width: int, height: int) -> list[tuple[int, int]]:
        """Return the four corner cells of a grid, the entry points of the walkers."""
        return [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    @staticmethod
    def _get_durability(namespace: SimpleNamespace) -> Expression:
        """Return the durability `T` of trails.

        It is the time over which the structure fades to bare ground.
        """
        durability: Expression = namespace.target.T
        return durability

    @staticmethod
    def _get_diffusion(namespace: SimpleNamespace) -> Expression:
        """Return the diffusion coefficient `D` of the trail potential."""
        coefficient: Expression = namespace.target.D
        return coefficient

    @staticmethod
    def _get_structure(namespace: SimpleNamespace) -> Expression:
        """Return the ground structure `G`, the goal of the trail potential."""
        structure: Expression = namespace.target.G
        return structure

    @staticmethod
    def _get_relaxation(namespace: SimpleNamespace) -> Expression:
        """Return the relaxation time `tau` of the trail potential."""
        time: Expression = namespace.target.tau
        return time

    def _get_goal(self, walker: AgentViewProtocol | None, system: SystemViewProtocol) -> dict[str, object]:
        """Return a random entry point other than the cell of a walker as its goal."""
        if not isinstance(walker, WalkerView):
            return {}
        here = cast('tuple[int, int]', walker.location)
        goals = [corner for corner in self._get_corners(round(walker.width), round(walker.height)) if corner != here]
        goal_x, goal_y = goals[int(system.rng.integers(len(goals)))]
        return {'goal_x': float(goal_x), 'goal_y': float(goal_y)}

    @staticmethod
    def _has_arrived(walker: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether a walker stands at its goal."""
        del system
        return isinstance(walker, WalkerView) and walker.location == (round(walker.goal_x), round(walker.goal_y))

    def _waits(self, walker: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether a walker arrived at its goal in the previous step.

        Such a walker leaves the green now.
        """
        return self._has_arrived(walker, system) and isinstance(walker, WalkerView) and bool(walker.arrived)

    def _walks(self, walker: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether a walker is away from its goal.

        Such a walker steps.
        """
        return not self._has_arrived(walker, system)

    def _get_arrived(self, walker: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return whether a walker stands at its goal at the end of its activation."""
        return float(self._has_arrived(walker, system))

    @staticmethod
    def _get_step(walker: AgentViewProtocol, system: SystemViewProtocol) -> object:
        """Return a neighbouring cell drawn with a weight growing with its alignment.

        The alignment is that of the step with the heading.

        The heading is the unit vector toward the goal plus `attraction` times the gradient of the trail potential, and
        the weight of a step is `exp(directedness * cos angle)`, where the angle is between the step and the heading.
        """
        if not isinstance(walker, WalkerView):
            return None
        x, y = cast('tuple[int, int]', walker.location)
        cells = cast('list[tuple[int, int]]', list(walker.get_neighbor_cells()))
        goal = np.array([walker.goal_x - x, walker.goal_y - y])
        steps = np.array([(cell[0] - x, cell[1] - y) for cell in cells], dtype=float)
        lengths = np.hypot(steps[:, 0], steps[:, 1])
        rises = np.array([system.get_field(TRAIL, cell) for cell in cells]) - system.get_field(TRAIL, (x, y))
        slope = 2 * (steps / lengths[:, None] ** 2).T @ rises / len(cells)
        heading = goal / max(float(np.hypot(*goal)), 1e-12) + walker.attraction * slope
        alignment = steps @ heading / (lengths * max(float(np.hypot(*heading)), 1e-12))
        weights = np.exp(walker.directedness * (alignment - alignment.max()))
        return cells[int(system.rng.choice(len(cells), p=weights / weights.sum()))]

    @staticmethod
    def _has_stepped(walker: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> bool:
        """Return whether a walker stepped in its activation.

        It steps since it had not arrived at its goal before.
        """
        del cell, system
        return isinstance(walker, WalkerView) and not walker.arrived

    @staticmethod
    def _get_footprint(walker: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the footprint a walker leaves on the ground of its cell.

        The footprint is `intensity (1 - G / G_max)`.
        """
        if not isinstance(walker, WalkerView):
            return 0.0
        return walker.intensity * (1 - system.get_field('target.G', cell) / walker.G_max)

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system."""
        rng = np.random.default_rng(self.random_state)
        corners = self._get_corners(self.size, self.size)
        starts = rng.integers(len(corners), size=self.n_walkers)
        goals = [corners[(start + 1 + int(rng.integers(len(corners) - 1))) % len(corners)] for start in starts]
        walkers = GroupElement(
            'walkers',
            element=WalkerElement,
            size=self.n_walkers,
            values={
                'goal_x': np.array([float(goal[0]) for goal in goals]),
                'goal_y': np.array([float(goal[1]) for goal in goals]),
                'attraction': self.attraction,
                'width': float(self.size),
                'height': float(self.size),
            },
            location=[corners[start] for start in starts],
        )
        ground = FieldElement(
            'ground',
            element=GroundElement,
            space=LatticeSpace((self.size, self.size), boundary='reflecting'),
            values={'T': self.T},
        )
        return CompositeSystem(
            self.label,
            systems=[
                CompositeSystem(
                    'people',
                    elements=[walkers],
                    space=LatticeSpace((self.size, self.size), boundary='fixed'),
                ),
                CompositeSystem(
                    'land',
                    elements=[ground],
                    interactions=[
                        RateInteraction(
                            'fading',
                            target='ground',
                            state='G',
                            rate=lambda s: -s.target.G / self._get_durability(s),
                        ),
                        RateInteraction(
                            'spreading',
                            target='ground',
                            state='V',
                            rate=lambda s: self._get_diffusion(s) * laplacian(s.target.V),
                        ),
                        RateInteraction(
                            'smoothing',
                            target='ground',
                            state='V',
                            rate=lambda s: (self._get_structure(s) - s.target.V) / self._get_relaxation(s),
                        ),
                    ],
                ),
            ],
            interactions=[
                MovementInteraction(
                    'walk',
                    target='people/walkers',
                    step=self._get_step,
                    condition=self._walks,
                    schedule='member',
                ),
                TransferInteraction(
                    'footprints',
                    source='people/walkers',
                    target='land/ground',
                    state=Source(),
                    into='G',
                    amount=self._get_footprint,
                    pairing='site',
                    guard=self._has_stepped,
                    form='agent',
                    schedule='member',
                ),
                BirthDeathInteraction(
                    'renewal',
                    target='people/walkers',
                    death=self._waits,
                    renew=True,
                    values=self._get_goal,
                    schedule='member',
                ),
                MapInteraction(
                    'arrival',
                    target='people/walkers',
                    state='arrived',
                    function=self._get_arrived,
                    form='agent',
                    schedule='member',
                ),
            ],
        )
