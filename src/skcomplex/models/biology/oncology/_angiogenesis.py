"""Oncology models: a tumour that grows on oxygen and recruits blood vessels by secreting growth factor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement, GroupElement
from ....core.laws import BaseLaw, RateLaw, RuleLaw
from ....core.laws.utils import laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._sprout_view import SproutView
from ._tumour_cell_view import TumourCellView
from ._vessel_view import VesselView

OXYGEN = 'tissue/tissue.oxygen'
VEGF = 'tissue/tissue.vegf'
TUMOUR = 'cells/tumour'
VESSELS = 'cells/vessels'


class TissueElement(Element):
    """Tissue in which oxygen and the vascular endothelial growth factor VEGF diffuse and decay.

    Both follow `u' = D Δu - lambda u`, and cells add or remove amounts at their sites: vessels supply oxygen, tumour
    cells consume it, and hypoxic tumour cells secrete VEGF.

    Args:
        label:
            The label of the tissue or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        oxygen:
            The initial oxygen.

        vegf:
            The initial VEGF.

        D_oxygen:
            The diffusion of oxygen.

        decay_oxygen:
            The decay rate of oxygen.

        D_vegf:
            The diffusion of VEGF.

        decay_vegf:
            The decay rate of VEGF.

    Examples:
        >>> from skcomplex.models.biology.oncology import TissueElement
        >>> TissueElement('tissue').D_vegf
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        oxygen: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        vegf: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D_oxygen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        decay_oxygen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        D_vegf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        decay_vegf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
    ) -> None:
        super().__init__(label, visual=visual)
        self.oxygen = oxygen
        self.vegf = vegf
        self.D_oxygen = D_oxygen
        self.decay_oxygen = decay_oxygen
        self.D_vegf = D_vegf
        self.decay_vegf = decay_vegf

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the diffusion and decay of oxygen and VEGF.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.oxygen, s.D_oxygen * laplacian(s.oxygen) - s.decay_oxygen * s.oxygen),
            RateLaw(s.vegf, s.D_vegf * laplacian(s.vegf) - s.decay_vegf * s.vegf),
        ]


class TumourCellElement(Element):
    """Tumour cell that consumes oxygen, divides when supplied, secretes VEGF when hypoxic, and dies without oxygen.

    Every step the cell takes up `consumption` oxygen at its site, or what is left. Above `proliferation` it divides
    with probability `division` into a free neighbouring cell. Below `hypoxia` it secretes `secretion` VEGF, and below
    `necrosis` it dies.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        hypoxic:
            Whether the cell is hypoxic, `1`, or not, `0`.

        consumption:
            The oxygen a cell takes up per step.

        proliferation:
            The oxygen above which a cell divides.

        division:
            The probability per step of a division.

        hypoxia:
            The oxygen below which a cell secretes VEGF.

        secretion:
            The VEGF a hypoxic cell secretes per step.

        necrosis:
            The oxygen below which a cell dies.

    Examples:
        >>> from skcomplex.models.biology.oncology import TumourCellElement
        >>> GroupElement('tumour', element=TumourCellElement, size=3, location=(5, 5)).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'hypoxic', 'color_range': (0.0, 1.0)}),
        *,
        hypoxic: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        consumption: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        proliferation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        division: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        hypoxia: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        secretion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        necrosis: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.hypoxic = hypoxic
        self.consumption = consumption
        self.proliferation = proliferation
        self.division = division
        self.hypoxia = hypoxia
        self.secretion = secretion
        self.necrosis = necrosis

    @classmethod
    def _get_view(cls) -> type[TumourCellView]:
        """Return the view of a tumour cell, `TumourCellView`."""
        return TumourCellView

    def _divide(self, agent: TumourCellView, model: ModelView) -> None:
        """Place a daughter cell in a random neighbouring cell without a tumour cell, if there is one."""
        taken = {other.location for other in agent.neighbors() if other.group == agent.group}
        free = [cell for cell in agent.neighbor_cells() if cell not in taken]
        if free:
            agent.create(location=free[int(model.rng.integers(len(free)))])

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the life cycle rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.live)]

    def live(self, agent: TumourCellView, model: ModelView) -> None:
        """Consume oxygen, then die, signal for vessels, or divide, by the oxygen at the cell.

        Args:
            agent:
                The tumour cell.

            model:
                The model.
        """
        cell = agent.location
        oxygen = model.field(OXYGEN, cell)
        model.deposit(OXYGEN, cell, -min(agent.consumption, oxygen))
        agent.hypoxic = float(oxygen < agent.hypoxia)
        if oxygen < agent.necrosis:
            agent.remove()
        elif agent.hypoxic:
            model.deposit(VEGF, cell, agent.secretion)
        elif oxygen > agent.proliferation and model.rng.random() < agent.division:
            self._divide(agent, model)


class VesselElement(Element):
    """Segment of a blood vessel, which supplies oxygen to its site every step.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        supply:
            The oxygen supplied per step.

    Examples:
        >>> from skcomplex.models.biology.oncology import VesselElement
        >>> GroupElement('vessels', element=VesselElement, size=2, location=[(0, 0), (0, 1)]).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        supply: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.supply = supply

    @classmethod
    def _get_view(cls) -> type[VesselView]:
        """Return the view of a vessel, `VesselView`."""
        return VesselView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the supply rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.supply_oxygen)]

    def supply_oxygen(self, agent: VesselView, model: ModelView) -> None:
        """Add oxygen at the segment's site.

        Args:
            agent:
                The vessel segment.

            model:
                The model.
        """
        model.deposit(OXYGEN, agent.location, agent.supply)


class SproutElement(Element):
    """Endothelial tip cell that climbs the VEGF gradient and leaves a new vessel segment behind it.

    Every step the tip moves to a neighbouring cell with a probability that grows with `exp(chemotaxis * VEGF /
    VEGF_here)`, so it senses the relative gradient, and it leaves a vessel segment where there was none. With
    probability `branching` the tip splits in two. Below the VEGF level `activation` the tip stays put.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        chemotaxis:
            How strongly VEGF draws the tip.

        branching:
            The probability per step of a new tip.

        activation:
            The VEGF above which the tip moves.

    Examples:
        >>> from skcomplex.models.biology.oncology import SproutElement
        >>> GroupElement('sprouts', element=SproutElement, size=2, location=(0, 5)).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        chemotaxis: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        branching: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        activation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.chemotaxis = chemotaxis
        self.branching = branching
        self.activation = activation

    @classmethod
    def _get_view(cls) -> type[SproutView]:
        """Return the view of a sprout tip, `SproutView`."""
        return SproutView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the sprouting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.sprout)]

    def sprout(self, agent: SproutView, model: ModelView) -> None:
        """Move up the VEGF gradient, leave a vessel segment, and branch now and then.

        Args:
            agent:
                The tip cell.

            model:
                The model.
        """
        here = agent.location
        level = model.field(VEGF, here)
        if level < agent.activation:
            return
        cells = cast('list[tuple[int, int]]', list(agent.neighbor_cells()))
        levels = np.array([model.field(VEGF, cell) for cell in cells])
        weights = np.exp(agent.chemotaxis * (levels - levels.max()) / level)
        if not agent.agents_here(VESSELS):
            model.create(VESSELS, location=here)
        agent.move_to(cells[model.rng.choice(len(cells), p=weights / weights.sum())])
        if model.rng.random() < agent.branching:
            agent.create(location=here)


@register_model(
    field='biology/oncology',
    tags=('agents', 'field', 'multiscale', 'angiogenesis', 'co-simulation'),
    references=('Carmichael and Hadžikadić 2019', 'Anderson and Chaplain 1998'),
    defaults={'until': 120, 'dt': 1},
)
def tumour_angiogenesis(
    width: int = 40,
    height: int = 25,
    *,
    n_sprouts: int = 5,
    radius: int = 2,
    chemotaxis: float = 30.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a tumour that outgrows its oxygen supply and recruits new blood vessels from a parent vessel.

    The scales are coupled: tumour cells and vessel tips are agents on a grid, and oxygen and VEGF are fields on a
    lattice of the same shape. The tumour starts far from the parent vessel along the left edge, so its cells become
    hypoxic and secrete VEGF, which draws tips out of the vessel. The new vessels bring oxygen and the tumour grows.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_sprouts:
            The number of tips on the parent vessel.

        radius:
            The radius of the initial tumour, centred three quarters of the way to the right.

        chemotaxis:
            How strongly VEGF draws the tips.

        random_state:
            The seed or generator of the positions of the tips.

    Returns:
        system:
            The model, with the agents `cells/tumour`, `cells/vessels`, and `cells/sprouts`, and the fields
            `tissue/tissue.oxygen` and `tissue/tissue.vegf`.

    Examples:
        >>> from skcomplex.models.biology.oncology import tumour_angiogenesis
        >>> [group.label for group in tumour_angiogenesis().systems[0].elements]
        ['tumour', 'vessels', 'sprouts']
    """
    rng = np.random.default_rng(random_state)
    center = (3 * width // 4, height // 2)
    tumour = [
        (x, y)
        for x in range(center[0] - radius, center[0] + radius + 1)
        for y in range(center[1] - radius, center[1] + radius + 1)
        if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    ]
    tips = sorted(int(row) for row in rng.choice(height, size=n_sprouts, replace=False))
    groups = [
        GroupElement('tumour', element=TumourCellElement, size=len(tumour), variable=True, location=tumour),
        GroupElement(
            'vessels',
            element=VesselElement,
            size=height,
            variable=True,
            location=[(0, row) for row in range(height)],
        ),
        GroupElement(
            'sprouts',
            element=SproutElement,
            size=n_sprouts,
            values={'chemotaxis': chemotaxis},
            variable=True,
            location=[(0, row) for row in tips],
        ),
    ]
    tissue = FieldElement('tissue', element=TissueElement, space=LatticeSpace((width, height), boundary='reflecting'))
    return System(
        systems=[
            System('cells', elements=groups, space=LatticeSpace((width, height), boundary='fixed')),
            System('tissue', elements=[tissue]),
        ],
    )
