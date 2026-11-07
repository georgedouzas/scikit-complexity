"""Fredkin's replicator: a planar parity automaton whose local law copies any pattern into translated replicas."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.components.elements import LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import (
    LatticeSpace,
    VonNeumannNeighborhood,
)
from ....core.components.system import System
from ..._catalog import register_system
from ._binary_cell import BinaryCellElement
from ._codes import life

HALF_CELL = 0.5


@register_system(
    field='computer_science/automata',
    tags=('lattice', 'replicator', 'linear', 'parity'),
    references=('Fredkin 1990', 'Dilão 1993', 'Dilão 2023'),
    defaults={'until': 32, 'dt': 1},
)
def fredkin_replicator(
    shape: Sequence[int] = (128, 128),
    *,
    initial: npt.ArrayLike | None = None,
    radius: float = 4.0,
) -> System:
    """Return Fredkin's parity automaton, whose local law copies any initial pattern into translated replicas.

    Every cell takes the sum modulo 2 of its four nearest neighbours, `x(i, j, n + 1) = x(i - 1, j, n) +
    x(i + 1, j, n) + x(i, j - 1, n) + x(i, j + 1, n) (mod 2)`. The law is linear over the integers modulo 2, and
    raising it to the power `2^k` gives the sum of the pattern translated by `2^k` cells in each of the four
    directions, so at the times `2^k` the initial pattern reappears four times, far enough apart not to overlap once
    `2^k` exceeds its width, and at other times the copies multiply in more complex arrangements. By default the
    pattern is a circumference, as in Dilão's example.

    Args:
        shape:
            The number of rows and columns of the torus.

        initial:
            The initial cells, an array of the shape, or `None` for a circumference of the radius in the middle.

        radius:
            The radius of the default circumference.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import fredkin_replicator
        >>> int(fredkin_replicator((32, 32)).elements[0].values['alive'].sum())
        32
    """
    if initial is None:
        rows, columns = np.indices(tuple(shape))
        distance = np.hypot(rows - shape[0] // 2, columns - shape[1] // 2)
        values = (np.abs(distance - radius) < HALF_CELL).astype(float)
    else:
        values = np.asarray(initial, dtype=float)
    space = LatticeSpace(tuple(shape))
    return System(
        elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': values})],
        interactions=[
            LocalRuleInteraction(
                'rule',
                target='cells',
                state='alive',
                rule=life((1, 3), (1, 3)),
                neighborhood=VonNeumannNeighborhood(),
            ),
        ],
    )
