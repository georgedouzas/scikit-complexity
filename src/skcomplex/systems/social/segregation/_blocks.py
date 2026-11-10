"""Block segregation systems: stochastic moves between blocks of a city.

The utility of their densities drives the moves.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ...general.agents import MovementInteraction

PHASE = 'default'


class DwellerView(AgentView):
    """View of a dweller of the stochastic Schelling system.

    Its states and parameters are attributes.
    """

    block: float
    beta: float
    altruism: float
    slope: float
    intercept: float


class DwellerElement(Element):
    """Agent of the stochastic Schelling system, with the utility `a * rho + b`.

    The utility depends on the density of its block.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        block:
            The index of the agent's block, which `StochasticSchellingSystem` sets after every move.

        beta:
            The inverse temperature of the moves.

        altruism:
            The weight of the change of the global utility against the agent's own utility.

        slope:
            The slope `a` of the utility in the block density.

        intercept:
            The intercept `b` of the utility.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.segregation import DwellerElement
        >>> group = GroupElement('dwellers', element=DwellerElement, size=10, values={'beta': 3.0})
        >>> group.label, group.element.__name__, group.size
        ('dwellers', 'DwellerElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        block: Annotated[float, StateSpec(domain=NN)] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        altruism: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        slope: Annotated[float, ParameterSpec()] = -4.0 / 3.0,
        intercept: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.block = block
        self.beta = beta
        self.altruism = altruism
        self.slope = slope
        self.intercept = intercept

    @classmethod
    def _get_view(cls) -> type[DwellerView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return DwellerView


class BlockMovesInteraction(MovementInteraction):
    """Moves of the stochastic Schelling system.

    A visited agent proposes to move to a random empty cell of the grid, and stays when the cell lies in its own block.
    With block densities `rho_i` before and `rho_f'` after the move, the proposal is accepted with probability
    `1 / (1 + exp(beta * dU))`, where `dU = (1 - altruism) * (u(rho_f') - u(rho_i)) + altruism * dU_global` and
    `dU_global` is the change of `M * sum_q rho_q u(rho_q)` over the blocks of `M` cells. The densities are the counts
    of the block states of the group. In `StochasticSchellingSystem`, a phase of visits draws as many agents with
    replacement as there are agents every step, and a `MapInteraction` sets the block of an agent after its move.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of dwellers.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        side:
            The side of a square block in cells, which divides the width and the height.

        probability:
            The probability that a visited agent proposes a move.

        phase:
            The phase of the step the moves run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.segregation import BlockMovesInteraction
        >>> BlockMovesInteraction('moves', target='dwellers', side=4).schedule
        'member'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        width: int = 20,
        height: int = 20,
        side: int = 5,
        probability: float = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            step=self._propose,
            probability=probability,
            accept=self._accept,
            schedule='member',
            phase=phase,
        )
        self.width = width
        self.height = height
        self.side = side

    def _propose(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> object:
        """Return a random empty cell of another block than the agent's, or `None` to stay."""
        if not isinstance(agent, DwellerView) or not isinstance(system, SystemView):
            return None
        cell = system.get_random_empty_cell()
        return (
            None if BlockMovesInteraction._get_block(cell, self.side, self.width // self.side) == agent.block else cell
        )

    def _accept(self, agent: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the probability `1 / (1 + exp(beta * dU))` of a move of an agent.

        The move is to a cell of another block.
        """
        if not isinstance(agent, DwellerView):
            return 0.0
        counts = system.get_composition('target', 'block')
        origin, destination = agent.block, float(
            BlockMovesInteraction._get_block(cell, self.side, self.width // self.side),
        )
        cells = self.side**2
        before = np.array([counts.get(origin, 0), counts.get(destination, 0)]) / cells
        after = before + np.array([-1, 1]) / cells
        slope, intercept = agent.slope, agent.intercept
        own = slope * (after[1] - before[0])
        total = cells * float(
            (after * (slope * after + intercept)).sum() - (before * (slope * before + intercept)).sum(),
        )
        change = (1 - agent.altruism) * own + agent.altruism * total
        return float(1 / (1 + np.exp(np.clip(agent.beta * change, -50, 50))))

    def get_block(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the index of the block that holds the cell of an agent.

        Args:
            agent:
                The dweller.

            system:
                The system view.

        Returns:
            block:
                The index of the block.
        """
        return float(BlockMovesInteraction._get_block(agent.location, self.side, self.width // self.side))

    @staticmethod
    def _get_block(location: object, side: int, columns: int) -> int:
        """Return the index of the block that holds a cell."""
        column, row = (int(item) for item in np.ravel(np.asarray(location)))
        return (row // side) * columns + column // side


class StochasticSchellingSystem(BaseSystem):
    """The stochastic Schelling system of agents moving between blocks.

    It segregates above `beta = 2`. The system has the group `dwellers`.

    Args:
        width:
            The number of columns, a multiple of the block side.

        height:
            The number of rows, a multiple of the block side.

        side:
            The side of a square block in cells.

        density:
            The share of cells with an agent.

        beta:
            The inverse temperature of the moves.

        altruism:
            The weight of the change of the global utility.

        slope:
            The slope of the utility in the block density.

        intercept:
            The intercept of the utility.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.segregation import StochasticSchellingSystem
        >>> StochasticSchellingSystem(8, 8, side=4, random_state=0).prepare().system_.elements[0].prepare().size_
        32
    """

    field = 'social/segregation'
    topics = ('segregation', 'agents', 'grid', 'stochastic')
    references = ('Grauwin et al. 2009', 'Jensen 2023')
    until = 50

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        side: int = 5,
        density: float = 0.5,
        beta: float = 3.0,
        altruism: float = 0.5,
        slope: float = -4.0 / 3.0,
        intercept: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.side = side
        self.density = density
        self.beta = beta
        self.altruism = altruism
        self.slope = slope
        self.intercept = intercept
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        size = int(self.width * self.height * self.density)
        cells = rng.permutation(self.width * self.height)[:size]
        locations = [(int(cell % self.width), int(cell // self.width)) for cell in cells]
        blocks = [
            BlockMovesInteraction._get_block(location, self.side, self.width // self.side) for location in locations
        ]
        dwellers = GroupElement(
            'dwellers',
            element=DwellerElement,
            size=size,
            values={
                'block': np.array(blocks, dtype=float),
                'beta': self.beta,
                'altruism': self.altruism,
                'slope': self.slope,
                'intercept': self.intercept,
            },
            location=locations,
        )
        moves = BlockMovesInteraction(
            'moves',
            target='dwellers',
            width=self.width,
            height=self.height,
            side=self.side,
            phase=PHASE,
        )
        block = MapInteraction(
            'block',
            target='dwellers',
            state='block',
            function=moves.get_block,
            schedule='member',
            form='agent',
            phase=PHASE,
        )
        return CompositeSystem(
            self.label,
            elements=[dwellers],
            interactions=[moves, block],
            space=LatticeSpace((self.width, self.height), boundary='fixed', capacity=1),
            time=DiscreteTime(
                schedule=PhasesScheduleSpec([PhaseSpec(PHASE, activation='visits', groups=('dwellers',))]),
            ),
        )
