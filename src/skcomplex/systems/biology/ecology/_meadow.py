"""Ecology systems: cows that graze a meadow of regrowing grass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._cow import CowElement
from ._grazing import GrazingInteraction
from ._herding import HerdingInteraction
from ._pasture import PastureElement


@register_system(
    field='biology/ecology',
    tags=('carrying capacity', 'agents', 'grid', 'grazing', 'commons'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 25},
)
def meadow_capacity(
    width: int = 40,
    height: int = 40,
    *,
    cows: int = 80,
    growth: float = 0.2,
    bite: float = 0.75,
    threshold: float = 0.9,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return cows that graze a summer of 25 weeks on a meadow, which shows how many cows the meadow can carry.

    Every week the grass regrows logistically, each cow eats `bite` of the grass of its patch, and each cow moves to the
    nearest patch that no other cow occupies and whose grass reaches `threshold`. A cow that finds none dies. Without
    regrowth a cow grazes its first patch and 25 more over the summer, so the meadow feeds about
    `width * height / 26` cows, and regrowth raises its carrying capacity.
    The cows graze the patches down at the same pace, so a herd above the capacity runs out of grown patches all at
    once and dies out within a week, instead of shrinking to the capacity.

    Args:
        width:
            The number of columns of the meadow.

        height:
            The number of rows of the meadow.

        cows:
            The initial number of cows.

        growth:
            The weekly growth rate of the grass.

        bite:
            The grass a cow eats in a week.

        threshold:
            The least grass a patch needs for a cow to move there.

        random_state:
            The seed or generator of the initial cells of the cows.

    Returns:
        system:
            The system, with the groups `pasture` and `cows`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import meadow_capacity
        >>> [group.size for group in meadow_capacity(10, 10, cows=5).elements]
        [100, 5]
    """
    rng = np.random.default_rng(random_state)
    cells = [(column, row) for column in range(width) for row in range(height)]
    picks = rng.choice(len(cells), size=cows, replace=False)
    return System(
        elements=[
            GroupElement(
                'pasture',
                element=PastureElement,
                size=len(cells),
                values={'growth': growth},
                variable=True,
                location=cells,
            ),
            GroupElement(
                'cows',
                element=CowElement,
                size=cows,
                values={'bite': bite},
                variable=True,
                location=[cells[int(pick)] for pick in picks],
            ),
        ],
        interactions=[
            GrazingInteraction('grazing', cows='cows', pasture='pasture'),
            HerdingInteraction('herding', cows='cows', pasture='pasture', threshold=threshold),
        ],
        space=LatticeSpace((width, height), boundary='fixed'),
    )
