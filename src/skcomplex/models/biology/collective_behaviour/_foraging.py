"""Collective behaviour models: animals that coordinate through signals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement, GroupElement
from ....core.interactions import DiffusionInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.stigmergy import EvaporationInteraction
from ._ant_view import AntView

PHEROMONE = 'ground/pheromone.level'


def _get_diffusion(namespace: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `D` of the pheromone."""
    coefficient: Expression = namespace.target.D
    return coefficient


class PheromoneElement(Element):
    """Pheromone of one cell of the ground, a level that diffuses to the neighbouring cells and evaporates.

    The element holds the level and its parameters. The model adds the diffusion `D laplacian(level)` and the
    evaporation `-evaporation level` as interactions.

    Args:
        label:
            The label of the element or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial level of pheromone.

        D:
            The diffusion coefficient of the pheromone.

        evaporation:
            The rate at which the pheromone evaporates.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import PheromoneElement
        >>> PheromoneElement('pheromone').evaporation
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        evaporation: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.D = D
        self.evaporation = evaporation


class AntElement(Element):
    """Ant that searches for food guided by pheromone and carries food home, marking the way with pheromone.

    A searching ant moves to a neighbouring cell with a probability that grows with its pheromone and, by `outward`,
    with its distance from the nest, as a scent of the nest would, and picks up food within a radius of the food. A
    carrying ant walks straight home and deposits pheromone on every cell it enters.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.collective_behaviour import AntElement
        >>> GroupElement('ants', element=AntElement, size=5, location=(0, 0)).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'carrying', 'color_range': (0.0, 1.0)}),
        *,
        carrying: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        nest_x: Annotated[float, ParameterSpec()] = 5.0,
        nest_y: Annotated[float, ParameterSpec()] = 10.0,
        food_x: Annotated[float, ParameterSpec()] = 34.0,
        food_y: Annotated[float, ParameterSpec()] = 10.0,
        food_radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        attraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        outward: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        deposit: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.carrying = carrying
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the foraging rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.forage)]

    def forage(self, agent: AntView, model: ModelView) -> None:
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
    ants = GroupElement(
        'ants',
        element=AntElement,
        size=n_ants,
        values={
            'nest_x': float(nest[0]),
            'nest_y': float(nest[1]),
            'food_x': float(food[0]),
            'food_y': float(food[1]),
            'attraction': attraction,
        },
        location=nest,
    )
    pheromone = FieldElement(
        'pheromone',
        element=PheromoneElement,
        space=LatticeSpace((width, height)),
        boundary={'level': 'zero-flux'},
        values={'D': D, 'evaporation': evaporation},
    )
    return System(
        systems=[
            System('colony', elements=[ants], space=LatticeSpace((width, height), boundary='fixed')),
            System(
                'ground',
                elements=[pheromone],
                interactions=[
                    DiffusionInteraction('diffusion', target='pheromone', state='level', coefficient=_get_diffusion),
                    EvaporationInteraction('evaporation', target='pheromone', state='level'),
                ],
            ),
        ],
    )
