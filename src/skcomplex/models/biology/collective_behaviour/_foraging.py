"""Collective behaviour models: animals that coordinate through signals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, cast

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import (
    AgentType,
    Element,
    Field,
    GridSpace,
    LatticeSpace,
    Law,
    Rate,
    Rule,
    Symbols,
    System,
    laplacian,
)
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

PHEROMONE = 'ground/pheromone.level'

Pheromone = Element.define(
    'Pheromone',
    states={'level': state(default=0.0, domain=Domain.NONNEGATIVE)},
    parameters={
        'D': parameter(default=0.05, domain=Domain.NONNEGATIVE),
        'evaporation': parameter(default=0.05, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [Rate(s.level, s.D * laplacian(s.level) - s.evaporation * s.level)],
)


class Ant(AgentType):
    """Ant that searches for food guided by pheromone and carries food home, marking the way with pheromone.

    A searching ant moves to a neighbouring cell with a probability that grows with its pheromone and, by `outward`,
    with its distance from the nest, as a scent of the nest would, and picks up food within a radius of the food. A
    carrying ant walks straight home and deposits pheromone on every cell it enters.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or one cell per ant.

        carrying:
            Whether the ant carries food.

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

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import Ant
        >>> Ant.group('ants', 5, location=(0, 0)).size
        5
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'carrying', 'color_range': (0.0, 1.0)})
    carrying: float = state(default=0.0, domain=Domain.BOOLEAN)
    nest_x: float = parameter(default=5.0)
    nest_y: float = parameter(default=10.0)
    food_x: float = parameter(default=34.0)
    food_y: float = parameter(default=10.0)
    food_radius: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    attraction: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    outward: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    deposit: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the foraging rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.forage)]

    def forage(self, agent: AgentView, model: ModelView) -> None:
        """Search for food along pheromone, or carry food home and mark the way.

        Args:
            agent:
                The ant.

            model:
                The model.
        """
        x, y = cast('tuple[int, int]', agent.location)
        nest = (round(agent.nest_x), round(agent.nest_y))
        if agent.carrying:
            if (x, y) == nest:
                agent.carrying = 0.0
                return
            home = (x + int(np.sign(nest[0] - x)), y + int(np.sign(nest[1] - y)))
            agent.move_to(home)
            model.deposit(PHEROMONE, home, agent.deposit)
            return
        if np.hypot(x - agent.food_x, y - agent.food_y) <= agent.food_radius:
            agent.carrying = 1.0
            return
        cells = cast('list[tuple[int, int]]', list(agent.neighbor_cells()))
        distance = np.hypot(x - nest[0], y - nest[1])
        weights = np.asarray(
            [
                (1.0 + agent.attraction * model.field(PHEROMONE, cell))
                * np.exp(agent.outward * (np.hypot(cell[0] - nest[0], cell[1] - nest[1]) - distance))
                for cell in cells
            ],
        )
        agent.move_to(cells[model.rng.choice(len(cells), p=weights / weights.sum())])


@register_model(
    field='biology/collective_behaviour',
    tags=('field', 'agents', 'stigmergy', 'co-simulation'),
    references=('Argun et al. 2021', 'Carmichael and Hadžikadić 2019'),
    defaults={'until': 600, 'dt': 1},
)
def pheromone_foraging(
    width: int = 40,
    height: int = 20,
    *,
    n_ants: int = 40,
    D: float = 0.05,
    evaporation: float = 0.05,
    attraction: float = 5.0,
) -> System:
    """Return ants that find food and lay a pheromone trail home, which other ants follow: stigmergy.

    The ants live on a grid and the pheromone on a lattice of the same shape, as two partitions of a co-simulation
    that exchange the field and the deposits at every step. The nest is near the left edge and the food near the right.

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

    Returns:
        system:
            The model, with the ants `colony/ants` and the pheromone `ground/pheromone.level`.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import pheromone_foraging
        >>> [child.label for child in pheromone_foraging().systems]
        ['colony', 'ground']
    """
    nest, food = (5, height // 2), (width - 6, height // 2)
    ants = Ant.group(
        'ants',
        n_ants,
        location=nest,
        nest_x=float(nest[0]),
        nest_y=float(nest[1]),
        food_x=float(food[0]),
        food_y=float(food[1]),
        attraction=attraction,
    )
    pheromone = Field(
        'pheromone',
        Pheromone,
        LatticeSpace((width, height)),
        boundary={'level': 'zero-flux'},
        values={'D': D, 'evaporation': evaporation},
    )
    return System(
        systems=[
            System('colony', elements=[ants], space=GridSpace(width, height)),
            System('ground', elements=[pheromone]),
        ],
    )
