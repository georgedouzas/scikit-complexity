"""Moves of the stochastic Schelling system, from an agent's cell to a random empty cell of another block."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.components.interactions import MovementInteraction
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ._dweller_view import DwellerView


class BlockMovesInteraction(MovementInteraction):
    """Moves of the stochastic Schelling system, from an agent's cell to a random empty cell of another block.

    A visited agent proposes to move to a random empty cell of the grid, and stays when the cell lies in its own block.
    With block densities `rho_i` before and `rho_f'` after the move, the proposal is accepted with probability
    `1 / (1 + exp(beta * dU))`, where `dU = (1 - altruism) * (u(rho_f') - u(rho_i)) + altruism * dU_global` and
    `dU_global` is the change of `M * sum_q rho_q u(rho_q)` over the blocks of `M` cells. The densities are the counts
    of the block states of the group. In `stochastic_schelling`, a phase of visits draws as many agents with
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
        cell = system.random_empty_cell()
        return (
            None if BlockMovesInteraction._block_of(cell, self.side, self.width // self.side) == agent.block else cell
        )

    def _accept(self, agent: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the probability `1 / (1 + exp(beta * dU))` of a move of an agent to a cell of another block."""
        if not isinstance(agent, DwellerView):
            return 0.0
        counts = system.composition('target', 'block')
        origin, destination = agent.block, float(
            BlockMovesInteraction._block_of(cell, self.side, self.width // self.side),
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
        return float(BlockMovesInteraction._block_of(agent.location, self.side, self.width // self.side))

    @staticmethod
    def _block_of(location: object, side: int, columns: int) -> int:
        """Return the index of the block that holds a cell."""
        column, row = (int(item) for item in np.ravel(np.asarray(location)))
        return (row // side) * columns + column // side
