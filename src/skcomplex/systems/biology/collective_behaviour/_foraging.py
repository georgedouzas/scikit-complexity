"""Animals that coordinate through signals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ...general.agents import MovementInteraction
from ..common.stigmergy import EvaporationInteraction

PHEROMONE = 'ground/pheromone.level'


class PheromoneElement(Element):
    """Pheromone of one cell of the ground.

    It is a level that diffuses to the neighbouring cells and evaporates.

    The element holds the level and its parameters. The system adds the diffusion `D laplacian(level)` and the
    evaporation `-evaporation level` as interactions.

    Args:
        label:
            The label of the element or the lattice field.

        level:
            The initial level of pheromone.

        D:
            The diffusion coefficient of the pheromone.

        evaporation:
            The rate at which the pheromone evaporates.

        space:
            The lattice whose cells hold the pheromone, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import PheromoneElement
        >>> PheromoneElement('pheromone').evaporation
        0.05
    """

    def __init__(
        self,
        label: str,
        *,
        level: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        D: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        evaporation: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.level = level
        self.D = D
        self.evaporation = evaporation


class AntView(AgentView):
    """View of an ant, with its states and parameters as attributes.

    `AntElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    carrying: float
    arrived: float
    nest_x: float
    nest_y: float
    food_x: float
    food_y: float
    food_radius: float
    attraction: float
    outward: float
    deposit: float


class AntElement(Element):
    """Ant that searches for food guided by pheromone and carries food home.

    It marks the way home with pheromone.

    A searching ant moves to a neighbouring cell with a probability that grows with its pheromone and, by `outward`,
    with its distance from the nest, as a scent of the nest would, and picks up food within a radius of the food. A
    carrying ant walks straight home, deposits pheromone on every cell it enters, and drops the food at the nest. The
    system adds the walk home, the deposit, the search, the pick-up and drop, and the arrival as member interactions,
    and an ant picks up or drops food the step after it arrives.

    Args:
        label:
            The label of the agent or group.

        carrying:
            Whether the ant carries food.

        arrived:
            Whether the ant stood at the nest with food, or near the food without it, at the end of the previous step.

        nest_x:
            The column of the nest.

        nest_y:
            The row of the nest.

        food_x:
            The column of the food.

        food_y:
            The row of the food.

        food_radius:
            The distance from the food within which ants pick it up.

        attraction:
            How strongly pheromone draws searching ants.

        outward:
            How strongly searching ants move away from the nest.

        deposit:
            The pheromone an ant deposits per step home.

        values:
            Further values by name, such as the cells `x` and `y` of the ants, or `None`.

        size:
            The number of ants, or `None` for one.

        space:
            The grid the ants walk on, whose axes `x` and `y` place them, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import AntElement
        >>> AntElement('ants', size=5).size
        5
    """

    visual = VisualSpec('agents', {'color_by': 'carrying', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        carrying: Annotated[float, FiniteEnumeratedSet([0, 1])] = 0.0,
        arrived: Annotated[float, FiniteEnumeratedSet([0, 1])] = 0.0,
        nest_x: float = 5.0,
        nest_y: float = 10.0,
        food_x: float = 34.0,
        food_y: float = 10.0,
        food_radius: Annotated[float, RealSet.closed_open(0, oo)] = 3.0,
        attraction: Annotated[float, RealSet.closed_open(0, oo)] = 5.0,
        outward: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        deposit: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.carrying = carrying
        self.arrived = arrived
        self.nest_x = nest_x
        self.nest_y = nest_y
        self.food_x = food_x
        self.food_y = food_y
        self.food_radius = food_radius
        self.attraction = attraction
        self.outward = outward
        self.deposit = deposit

    @classmethod
    def _get_view(cls) -> type[AntView]:
        """Return the view of an ant, `AntView`."""
        return AntView


class PheromoneForagingSystem(BaseSystem):
    """Ants that find food and lay a pheromone trail home, which other ants follow.

    It is stigmergy.

    The ants live on a grid and the pheromone on a lattice of the same shape, as two partitions of a co-simulation that
    exchange the field and the deposits at every step. The nest is near the left edge and the food near the right. The
    system has the ants `colony/ants` and the pheromone `ground/pheromone.level`.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_ants:
            The number of ants, which start at the nest.

        D:
            The diffusion of the pheromone.

        evaporation:
            The evaporation rate of the pheromone.

        attraction:
            How strongly pheromone draws searching ants.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import PheromoneForagingSystem
        >>> [child.label for child in PheromoneForagingSystem().prepare().system_.systems]
        ['colony', 'ground']
    """

    field = 'biology/collective_behaviour'
    topics = ('field', 'agents', 'stigmergy', 'co-simulation')
    references = ('Argun et al. 2021', 'Carmichael and Hadžikadić 2019')
    until = 600
    dt = 1

    def __init__(
        self,
        width: int = 40,
        height: int = 20,
        *,
        n_ants: int = 40,
        D: float = 0.05,
        evaporation: float = 0.05,
        attraction: float = 5.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.n_ants = n_ants
        self.D = D
        self.evaporation = evaporation
        self.attraction = attraction

    @staticmethod
    def _get_diffusion(namespace: SimpleNamespace) -> Expression:
        """Return the diffusion coefficient `D` of the pheromone."""
        coefficient: Expression = namespace.target.D
        return coefficient

    @staticmethod
    def _get_nest(ant: AntView) -> tuple[int, int]:
        """Return the cell of the nest of an ant."""
        return round(ant.nest_x), round(ant.nest_y)

    def _is_homing(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an ant carries food away from the nest.

        Such an ant walks home.
        """
        del system
        return isinstance(ant, AntView) and bool(ant.carrying) and ant.location != self._get_nest(ant)

    @staticmethod
    def _searches(ant: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an ant without food stands farther than `food_radius` from the food.

        Such an ant searches.
        """
        del system
        if not isinstance(ant, AntView) or ant.carrying:
            return False
        x, y = cast('tuple[int, int]', ant.location)
        return bool(np.hypot(x - ant.food_x, y - ant.food_y) > ant.food_radius)

    def _get_home_step(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> object:
        """Return the cell one step straight toward the nest."""
        del system
        if not isinstance(ant, AntView):
            return None
        x, y = cast('tuple[int, int]', ant.location)
        nest = self._get_nest(ant)
        return x + int(np.sign(nest[0] - x)), y + int(np.sign(nest[1] - y))

    def _get_search_step(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> object:
        """Return a neighbouring cell drawn with a weight growing with its pheromone.

        The weight also grows with the distance of the cell from the nest.
        """
        if not isinstance(ant, AntView):
            return None
        x, y = cast('tuple[int, int]', ant.location)
        nest = self._get_nest(ant)
        cells = cast('list[tuple[int, int]]', list(ant.get_neighbor_cells()))
        distance = np.hypot(x - nest[0], y - nest[1])
        weights = np.asarray(
            [
                (1.0 + ant.attraction * system.get_field(PHEROMONE, cell))
                * np.exp(ant.outward * (np.hypot(cell[0] - nest[0], cell[1] - nest[1]) - distance))
                for cell in cells
            ],
        )
        return cells[int(system.rng.choice(len(cells), p=weights / weights.sum()))]

    @staticmethod
    def _carries(ant: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> bool:
        """Return whether an ant carries food and walked home in its activation.

        Such an ant marks its cell with pheromone.
        """
        del cell, system
        return isinstance(ant, AntView) and bool(ant.carrying) and not ant.arrived

    @staticmethod
    def _get_deposit(ant: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the pheromone `deposit` an ant leaves on its cell."""
        del cell, system
        return float(ant.deposit) if isinstance(ant, AntView) else 0.0

    def _mark(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Leave the pheromone of an ant that carries food on the ground of its cell."""
        cell = ant.location_key
        if not ant.alive or not self._carries(ant, cell, system):
            return
        system.deposit('pheromone.level', cell, self._get_deposit(ant, cell, system))

    @staticmethod
    def _get_carrying(ant: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return whether an ant carries food next.

        The step after arriving, it takes food or drops it at the nest.
        """
        if not isinstance(ant, AntView):
            return 0.0
        return float(not ant.carrying) if ant.arrived else float(ant.carrying)

    def _get_arrived(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return whether an ant arrived at the end of its turn.

        It arrived when it stands at the nest with food, or near the food without it.
        """
        if not isinstance(ant, AntView):
            return 0.0
        return float(not self._is_homing(ant, system) and not self._searches(ant, system))

    def _carry(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Let an ant take food at the food source or drop it at the nest, the step after it arrived."""
        if isinstance(ant, AntView):
            ant.carrying = self._get_carrying(ant, system)

    def _arrive(self, ant: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Mark whether an ant arrived at the end of its turn, at the nest with food or near the food without it."""
        if isinstance(ant, AntView):
            ant.arrived = self._get_arrived(ant, system)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of ants that find food and lay a pheromone trail home.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        nest, food = (5, self.height // 2), (self.width - 6, self.height // 2)
        grid = LatticeSpace((self.width, self.height), boundary='fixed')
        ants = AntElement(
            'ants',
            nest_x=float(nest[0]),
            nest_y=float(nest[1]),
            food_x=float(food[0]),
            food_y=float(food[1]),
            attraction=self.attraction,
            values={'x': np.full(self.n_ants, nest[0]), 'y': np.full(self.n_ants, nest[1])},
            size=self.n_ants,
            space=grid,
        )
        pheromone = PheromoneElement(
            'pheromone',
            D=self.D,
            evaporation=self.evaporation,
            space=LatticeSpace((self.width, self.height), boundary={'level': 'zero-flux'}),
        )
        self.system_ = System(
            self.label,
            systems=[
                System('colony', elements=[ants], space=grid),
                System(
                    'ground',
                    elements=[pheromone],
                    interactions=[
                        Interaction(
                            'diffusion',
                            roles={'target': 'pheromone'},
                            laws=lambda s: [
                                RateLaw(s.target.level, self._get_diffusion(s) * laplacian(s.target.level)),
                            ],
                        ),
                        EvaporationInteraction('evaporation', target='pheromone', state='level'),
                    ],
                ),
            ],
            interactions=[
                MovementInteraction(
                    'homing',
                    target='colony/ants',
                    step=self._get_home_step,
                    condition=self._is_homing,
                    schedule='member',
                ),
                Interaction(
                    'marking',
                    roles={'target': 'colony/ants', 'pheromone': 'ground/pheromone'},
                    laws=lambda s: [UpdateLaw(s.target, self._mark)],
                ),
                MovementInteraction(
                    'search',
                    target='colony/ants',
                    step=self._get_search_step,
                    condition=self._searches,
                    schedule='member',
                ),
                Interaction(
                    'carrying',
                    roles={'target': 'colony/ants'},
                    laws=lambda s: [UpdateLaw(s.target, self._carry)],
                ),
                Interaction(
                    'arrival',
                    roles={'target': 'colony/ants'},
                    laws=lambda s: [UpdateLaw(s.target, self._arrive)],
                ),
            ],
        )
