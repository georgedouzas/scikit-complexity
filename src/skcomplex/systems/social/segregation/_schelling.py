"""Segregation systems: residential sorting of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._relocation import RelocationInteraction
from ._resident import ResidentElement


@register_system(
    field='social/segregation',
    tags=('segregation', 'agents', 'grid'),
    references=('Schelling 1971',),
    defaults={'until': 50},
)
def schelling(
    width: int = 20,
    height: int = 20,
    *,
    density: float = 0.8,
    minority: float = 0.3,
    threshold: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a Schelling segregation system on a grid with at most one resident per cell.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        density:
            The share of cells with a resident.

        minority:
            The share of residents of kind 1.

        threshold:
            The smallest share of similar neighbours that keeps a resident in place.

        random_state:
            The seed or generator of the kinds.

    Returns:
        system:
            The system, with the group `residents`.

    Examples:
        >>> from skcomplex.systems.social.segregation import schelling
        >>> schelling(10, 10, density=0.5).elements[0].prepare().size_
        50
    """
    rng = np.random.default_rng(random_state)
    size = int(width * height * density)
    kinds = (rng.random(size) < minority).astype(float)
    residents = GroupElement(
        'residents',
        element=ResidentElement,
        size=size,
        values={'kind': kinds, 'threshold': threshold},
    )
    return System(
        elements=[residents],
        interactions=[RelocationInteraction('relocation', target='residents')],
        space=LatticeSpace((width, height), boundary='fixed', capacity=1),
    )
