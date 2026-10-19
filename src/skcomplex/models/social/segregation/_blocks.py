"""Block segregation models: stochastic moves between blocks of a city driven by the utility of their densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, role, state
from ....core.model import AgentType, GridSpace, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]


def _block_of(location: object, side: int, columns: int) -> int:
    """Return the index of the block that holds a cell."""
    column, row = (int(item) for item in np.ravel(np.asarray(location)))
    return (row // side) * columns + column // side


class Dweller(AgentType):
    """Agent of the stochastic Schelling model, whose utility `a * rho + b` depends on the density of its block.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random empty cell.

        block:
            The index of the agent's block, set by the moves.

        beta:
            The inverse temperature of the moves.

        altruism:
            The weight of the change of the global utility against the agent's own utility.

        slope:
            The slope `a` of the utility in the block density.

        intercept:
            The intercept `b` of the utility.

    Examples:
        >>> from skcomplex.models.social.segregation import Dweller
        >>> Dweller.group('dwellers', 10, beta=3.0)
        Group('dwellers', element=Dweller, size=10)
    """

    block: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    beta: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    altruism: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    slope: float = parameter(default=-4.0 / 3.0)
    intercept: float = parameter(default=1.0)


class BlockMoves(Interaction):
    """Moves of the stochastic Schelling model, as many proposals per step as there are agents.

    A random agent proposes to move to a random empty cell of another block. With block densities `rho_i` before and
    `rho_f'` after the move, the proposal is accepted with probability `1 / (1 + exp(beta * dU))`, where
    `dU = (1 - altruism) * (u(rho_f') - u(rho_i)) + altruism * dU_global` and `dU_global` is the change of
    `M * sum_q rho_q u(rho_q)` over the blocks of `M` cells.

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

    Examples:
        >>> from skcomplex.models.social.segregation import BlockMoves
        >>> BlockMoves('moves', target='dwellers', side=4).side
        4
    """

    target: str = role()
    width: int = 20
    height: int = 20
    side: int = 5

    def laws(self, s: Symbols) -> list[Law]:
        """Return the move rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.move)]

    def _change(self, agent: AgentView, counts: Array, origin: int, destination: int) -> float:
        """Return the utility change `dU` of a move between two blocks."""
        cells = self.side**2
        before = counts[[origin, destination]] / cells
        after = (counts[[origin, destination]] + [-1, 1]) / cells
        slope, intercept = agent.slope, agent.intercept
        own = slope * (after[1] - before[0])
        total = cells * float(
            (after * (slope * after + intercept)).sum() - (before * (slope * before + intercept)).sum(),
        )
        return float((1 - agent.altruism) * own + agent.altruism * total)

    def move(self, model: ModelView) -> None:
        """Run one move proposal per agent.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        columns = self.width // self.side
        counts = np.zeros(columns * (self.height // self.side))
        for agent in agents:
            counts[_block_of(agent.location, self.side, columns)] += 1
        for _ in range(len(agents)):
            agent = agents[int(model.rng.integers(len(agents)))]
            cell = model.random_empty_cell()
            origin, destination = _block_of(agent.location, self.side, columns), _block_of(cell, self.side, columns)
            if origin == destination:
                continue
            change = self._change(agent, counts, origin, destination)
            if model.rng.random() < 1 / (1 + np.exp(np.clip(agent.beta * change, -50, 50))):
                agent.move_to(cell)
                agent.block = float(destination)
                counts[[origin, destination]] += [-1, 1]


@register_model(
    field='social/segregation',
    tags=('segregation', 'agents', 'grid', 'stochastic'),
    references=('Grauwin et al. 2009', 'Jensen 2023'),
    defaults={'until': 50},
)
def stochastic_schelling(
    width: int = 20,
    height: int = 20,
    *,
    side: int = 5,
    density: float = 0.5,
    beta: float = 3.0,
    altruism: float = 0.5,
    slope: float = -4.0 / 3.0,
    intercept: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return the stochastic Schelling model of agents moving between blocks, which segregates above `beta = 2`.

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

    Returns:
        system:
            The model, with the group `dwellers`.

    Examples:
        >>> from skcomplex.models.social.segregation import stochastic_schelling
        >>> stochastic_schelling(8, 8, side=4, random_state=0).elements[0].size
        32
    """
    rng = np.random.default_rng(random_state)
    size = int(width * height * density)
    cells = rng.permutation(width * height)[:size]
    locations = [(int(cell % width), int(cell // width)) for cell in cells]
    blocks = [_block_of(location, side, width // side) for location in locations]
    dwellers = Dweller.group(
        'dwellers',
        size,
        location=locations,
        block=np.array(blocks, dtype=float),
        beta=beta,
        altruism=altruism,
        slope=slope,
        intercept=intercept,
    )
    moves = BlockMoves('moves', target='dwellers', width=width, height=height, side=side)
    return System(elements=[dwellers], interactions=[moves], space=GridSpace(width, height, capacity=1))
