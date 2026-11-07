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
    references=('Schelling 1971', 'Downey 2018'),
    defaults={'until': 50},
)
def schelling(
    width: int = 20,
    height: int = 20,
    *,
    density: float = 0.8,
    minority: float = 0.3,
    threshold: float = 0.5,
    choices: int | None = None,
    moving: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a Schelling segregation system on a grid with at most one resident per cell.

    Every step every resident whose share of neighbours of its own kind is below the threshold moves to a random empty
    cell. With `choices`, the Big Sort of Downey's Exercise 9-1 after Bishop's hypothesis, residents move for any
    reason, a share `moving` of them every step, and choose among `choices` random empty cells the one with the
    highest share of neighbours of their own kind, so the threshold plays no part.

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

        choices:
            The number of random empty cells a moving resident compares under the Big Sort, or `None` for Schelling's
            rule.

        moving:
            The probability that a resident moves in a step under the Big Sort.

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
        interactions=[
            (
                RelocationInteraction('relocation', target='residents')
                if choices is None
                else RelocationInteraction('relocation', target='residents', probability=moving, choices=choices)
            ),
        ],
        space=LatticeSpace((width, height), boundary='fixed', capacity=1),
    )
