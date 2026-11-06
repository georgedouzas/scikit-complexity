"""Oncology systems: a tumour that grows on oxygen and recruits blood vessels by secreting growth factor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import FieldElement, GroupElement
from ....core.components.interactions import (
    DiffusionInteraction,
    LifeCycleInteraction,
    MapInteraction,
    MovementInteraction,
    RateInteraction,
    TransferInteraction,
)
from ....core.components.laws import Source
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ..._catalog import register_system
from ._sprout import SproutElement
from ._sprout_view import SproutView
from ._tissue import TissueElement
from ._tumour_cell import TumourCellElement
from ._tumour_cell_view import TumourCellView
from ._vessel import VesselElement
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


def _get_uptake(cell: AgentViewProtocol, site: object, system: SystemViewProtocol) -> float:
    """Return minus the oxygen a tumour cell takes up at its site, `consumption` or what is left."""
    if not isinstance(cell, TumourCellView):
        return 0.0
    return -min(cell.consumption, system.field('target.oxygen', site))


def _get_hypoxic(cell: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return whether the oxygen at the site of a tumour cell is below `hypoxia`."""
    if not isinstance(cell, TumourCellView):
        return 0.0
    return float(system.field(OXYGEN, cell.location) < cell.hypoxia)


def _secretes(cell: AgentViewProtocol, site: object, system: SystemViewProtocol) -> bool:
    """Return whether a living tumour cell is hypoxic, so it secretes VEGF."""
    if not isinstance(cell, TumourCellView):
        return False
    return bool(cell.hypoxic) and system.field(OXYGEN, site) >= cell.necrosis


def _get_secretion(cell: AgentViewProtocol, site: object, system: SystemViewProtocol) -> float:
    """Return the VEGF `secretion` of a hypoxic tumour cell."""
    del site, system
    return float(cell.secretion) if isinstance(cell, TumourCellView) else 0.0


def _dies(cell: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether the oxygen at the site of a tumour cell is below `necrosis`, so it dies."""
    return isinstance(cell, TumourCellView) and system.field(OXYGEN, cell.location) < cell.necrosis


def _get_division(cell: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the probability `division` that a supplied tumour cell divides, above `proliferation` oxygen."""
    if not isinstance(cell, TumourCellView) or cell.hypoxic:
        return 0.0
    return float(cell.division) if system.field(OXYGEN, cell.location) > cell.proliferation else 0.0


def _get_supply(vessel: AgentViewProtocol, site: object, system: SystemViewProtocol) -> float:
    """Return the oxygen `supply` of a vessel segment."""
    del site, system
    return float(vessel.supply) if isinstance(vessel, VesselView) else 0.0


def _active(tip: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether the VEGF at a tip reaches `activation`, so the tip moves."""
    return isinstance(tip, SproutView) and system.field(VEGF, tip.location) >= tip.activation


def _lays(tip: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether an active tip stands where there is no vessel segment, so it leaves one."""
    return _active(tip, system) and not tip.agents_here(VESSELS)


def _get_branching(tip: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the probability `branching` that an active tip splits in two."""
    return float(tip.branching) if isinstance(tip, SproutView) and _active(tip, system) else 0.0


def _get_chemotaxis(tip: AgentViewProtocol, system: SystemViewProtocol) -> object:
    """Return a neighbouring cell drawn with a weight that grows with `exp(chemotaxis * VEGF / VEGF_here)`."""
    if not isinstance(tip, SproutView):
        return None
    level = system.field(VEGF, tip.location)
    cells = cast('list[tuple[int, int]]', list(tip.neighbor_cells()))
    levels = np.array([system.field(VEGF, cell) for cell in cells])
    weights = np.exp(tip.chemotaxis * (levels - levels.max()) / level)
    return cells[int(system.rng.choice(len(cells), p=weights / weights.sum()))]


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


@register_system(
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
            The system, with the agents `cells/tumour`, `cells/vessels`, and `cells/sprouts`, and the fields
            `tissue/tissue.oxygen` and `tissue/tissue.vegf`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import tumour_angiogenesis
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
