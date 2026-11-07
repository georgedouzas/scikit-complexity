"""Stochastic process systems: random iterations of interval maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ._random_branch_map import RandomBranchMapElement


@register_system(
    field='mathematics/stochastic_processes',
    tags=('random map', 'iterated function system', 'interval map', 'stationary distribution'),
    references=('Dilão 2023',),
    defaults={'until': 200, 'dt': 1},
)
def random_branch_map(
    n_orbits: int = 200,
    *,
    p1: float = 0.5,
    mu: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return orbits of a random map of `[0, 1]` with the two branches `(1 + x) / 2` and `mu x²`.

    At every step each orbit applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise (Dilão 2023, Exercise
    5.10). The first branch pulls towards 1 and the second towards 0, so the orbits settle on a stationary
    distribution whose shape changes with `p1` and `mu`. For `mu = 0` the second branch resets the orbit to 0, and
    the stationary mean is `p1 / (2 - p1)`. For `mu = 1` both branches fix `x = 1`, which attracts the orbits once
    `p1 > 1/2`, since there the first branch halves the distance to 1 and the second doubles it.

    Args:
        n_orbits:
            The number of independent orbits.

        p1:
            The probability of the branch `(1 + x) / 2`.

        mu:
            The coefficient of the branch `mu x²`, between 0 and 1.

        random_state:
            The seed or generator of the initial values, uniform on `[0, 1]`.

    Returns:
        system:
            The system, with the group `orbits` of state `x`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import random_branch_map
        >>> random_branch_map(10, p1=0.3).elements[0].prepare().size_
        10
    """
    start = np.random.default_rng(random_state).uniform(0.0, 1.0, n_orbits)
    return System(
        elements=[
            GroupElement(
                'orbits',
                element=RandomBranchMapElement,
                size=n_orbits,
                values={'x': start, 'p1': p1, 'mu': mu},
            ),
        ],
    )
