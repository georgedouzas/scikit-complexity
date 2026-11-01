"""Oncology models: a tumour that grows on oxygen and recruits blood vessels by secreting growth factor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement, GroupElement
from ....core.interactions import (
    DiffusionInteraction,
    LifeCycleInteraction,
    MapInteraction,
    MovementInteraction,
    RateInteraction,
    TransferInteraction,
)
from ....core.laws import Source
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._sprout_view import SproutView
from ._tumour_cell_view import TumourCellView
from ._vessel_view import VesselView

OXYGEN = 'tissue/tissue.oxygen'
VEGF = 'tissue/tissue.vegf'
VESSELS = 'cells/vessels'
TUMOUR = 'cells/tumour'
SPROUTS = 'cells/sprouts'
TISSUE = 'tissue/tissue'


def _get_oxygen_decay(namespace: SimpleNamespace) -> Expression:
    """Return the decay `-decay_oxygen oxygen` of oxygen."""
    rate: Expression = -namespace.target.decay_oxygen * namespace.target.oxygen
    return rate


def _get_oxygen_diffusion(namespace: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `D_oxygen` of oxygen."""
    coefficient: Expression = namespace.target.D_oxygen
    return coefficient


def _get_vegf_decay(namespace: SimpleNamespace) -> Expression:
    """Return the decay `-decay_vegf vegf` of VEGF."""
    rate: Expression = -namespace.target.decay_vegf * namespace.target.vegf
    return rate


def _get_vegf_diffusion(namespace: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `D_vegf` of VEGF."""
    coefficient: Expression = namespace.target.D_vegf
    return coefficient


def _get_uptake(cell: AgentViewProtocol, site: object, model: ModelViewProtocol) -> float:
    """Return minus the oxygen a tumour cell takes up at its site, `consumption` or what is left."""
    if not isinstance(cell, TumourCellView):
        return 0.0
    return -min(cell.consumption, model.field('target.oxygen', site))


def _get_hypoxic(cell: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return whether the oxygen at the site of a tumour cell is below `hypoxia`."""
    if not isinstance(cell, TumourCellView):
        return 0.0
    return float(model.field(OXYGEN, cell.location) < cell.hypoxia)


def _secretes(cell: AgentViewProtocol, site: object, model: ModelViewProtocol) -> bool:
    """Return whether a living tumour cell is hypoxic, so it secretes VEGF."""
    if not isinstance(cell, TumourCellView):
        return False
    return bool(cell.hypoxic) and model.field(OXYGEN, site) >= cell.necrosis


def _get_secretion(cell: AgentViewProtocol, site: object, model: ModelViewProtocol) -> float:
    """Return the VEGF `secretion` of a hypoxic tumour cell."""
    del site, model
    return float(cell.secretion) if isinstance(cell, TumourCellView) else 0.0


def _dies(cell: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether the oxygen at the site of a tumour cell is below `necrosis`, so it dies."""
    return isinstance(cell, TumourCellView) and model.field(OXYGEN, cell.location) < cell.necrosis


def _get_division(cell: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `division` that a supplied tumour cell divides, above `proliferation` oxygen."""
    if not isinstance(cell, TumourCellView) or cell.hypoxic:
        return 0.0
    return float(cell.division) if model.field(OXYGEN, cell.location) > cell.proliferation else 0.0


def _get_supply(vessel: AgentViewProtocol, site: object, model: ModelViewProtocol) -> float:
    """Return the oxygen `supply` of a vessel segment."""
    del site, model
    return float(vessel.supply) if isinstance(vessel, VesselView) else 0.0


def _active(tip: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether the VEGF at a tip reaches `activation`, so the tip moves."""
    return isinstance(tip, SproutView) and model.field(VEGF, tip.location) >= tip.activation


def _lays(tip: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether an active tip stands where there is no vessel segment, so it leaves one."""
    return _active(tip, model) and not tip.agents_here(VESSELS)


def _get_branching(tip: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `branching` that an active tip splits in two."""
    return float(tip.branching) if isinstance(tip, SproutView) and _active(tip, model) else 0.0


def _get_chemotaxis(tip: AgentViewProtocol, model: ModelViewProtocol) -> object:
    """Return a neighbouring cell drawn with a weight that grows with `exp(chemotaxis * VEGF / VEGF_here)`."""
    if not isinstance(tip, SproutView):
        return None
    level = model.field(VEGF, tip.location)
    cells = cast('list[tuple[int, int]]', list(tip.neighbor_cells()))
    levels = np.array([model.field(VEGF, cell) for cell in cells])
    weights = np.exp(tip.chemotaxis * (levels - levels.max()) / level)
    return cells[int(model.rng.choice(len(cells), p=weights / weights.sum()))]


def _get_tissue() -> list[DiffusionInteraction | RateInteraction]:
    """Return the diffusion and the decay of oxygen and VEGF in the tissue."""
    return [
        DiffusionInteraction('oxygen_diffusion', target='tissue', state='oxygen', coefficient=_get_oxygen_diffusion),
        RateInteraction('oxygen_decay', target='tissue', state='oxygen', rate=_get_oxygen_decay),
        DiffusionInteraction('vegf_diffusion', target='tissue', state='vegf', coefficient=_get_vegf_diffusion),
        RateInteraction('vegf_decay', target='tissue', state='vegf', rate=_get_vegf_decay),
    ]


def _get_cells() -> list[TransferInteraction | MapInteraction | LifeCycleInteraction | MovementInteraction]:
    """Return the member interactions of the tumour cells, the vessel segments, and the tips."""
    return [
        TransferInteraction(
            'uptake',
            source=TUMOUR,
            target=TISSUE,
            state=Source(),
            into='oxygen',
            amount=_get_uptake,
            pairing='site',
            form='agent',
            schedule='member',
        ),
        MapInteraction(
            'hypoxia',
            target=TUMOUR,
            state='hypoxic',
            function=_get_hypoxic,
            form='agent',
            schedule='member',
        ),
        TransferInteraction(
            'secretion',
            source=TUMOUR,
            target=TISSUE,
            into='vegf',
            state=Source(),
            amount=_get_secretion,
            pairing='site',
            guard=_secretes,
            form='agent',
            schedule='member',
        ),
        LifeCycleInteraction(
            'tumour_life',
            target=TUMOUR,
            death=_dies,
            birth=_get_division,
            placement='neighbor',
            schedule='member',
        ),
        TransferInteraction(
            'supply',
            source=VESSELS,
            target=TISSUE,
            state=Source(),
            into='oxygen',
            amount=_get_supply,
            pairing='site',
            form='agent',
            schedule='member',
        ),
        LifeCycleInteraction(
            'laying',
            target=SPROUTS,
            group=VESSELS,
            birth=_lays,
            schedule='member',
        ),
        LifeCycleInteraction('branching', target=SPROUTS, birth=_get_branching, schedule='member'),
        MovementInteraction('chemotaxis', target=SPROUTS, step=_get_chemotaxis, condition=_active, schedule='member'),
    ]


class TissueElement(Element):
    """Tissue in which oxygen and the vascular endothelial growth factor VEGF diffuse and decay.

    Both follow `u' = D Δu - lambda u`, and cells add or remove amounts at their sites: vessels supply oxygen, tumour
    cells consume it, and hypoxic tumour cells secrete VEGF. The model adds the diffusion and the decay as
    interactions.

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


class TumourCellElement(Element):
    """Tumour cell that consumes oxygen, divides when supplied, secretes VEGF when hypoxic, and dies without oxygen.

    Every step the cell takes up `consumption` oxygen at its site, or what is left. Above `proliferation` it divides
    with probability `division` into a free neighbouring cell. Below `hypoxia` it secretes `secretion` VEGF, and below
    `necrosis` it dies. The model adds the uptake, the hypoxia, the secretion, and the life cycle as member
    interactions.

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


class VesselElement(Element):
    """Segment of a blood vessel, which supplies oxygen to its site every step, a member transfer of the model.

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


class SproutElement(Element):
    """Endothelial tip cell that climbs the VEGF gradient and leaves a new vessel segment behind it.

    Every step the tip moves to a neighbouring cell with a probability that grows with `exp(chemotaxis * VEGF /
    VEGF_here)`, so it senses the relative gradient, and it leaves a vessel segment where there was none. With
    probability `branching` the tip splits in two, the new tip standing where the tip was. Below the VEGF level
    `activation` the tip stays put. The model adds the vessel segment, the branching, and the step as member
    interactions.

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
            System('tissue', elements=[tissue], interactions=_get_tissue()),
        ],
        interactions=_get_cells(),
    )
