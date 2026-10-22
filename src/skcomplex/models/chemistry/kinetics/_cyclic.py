"""Cyclic cellular automaton of excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, LocalRule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace, Neighborhood
from ....core.system import System

type Array = npt.NDArray[np.float64]


def _advance_cyclically(level: Array, *around: Array, n: Any, threshold: Any) -> Array:  # noqa: ANN401
    """Return the next levels, one step around the cycle where enough neighbours hold the following level."""
    following = np.mod(level + 1, n)
    successors = sum((item == following).astype(float) for item in around)
    return np.where(successors >= threshold, following, level)


class CyclicCell(Element):
    """Cell of Griffeath's cyclic cellular automaton, with `n` levels that follow each other in a cycle.

    A cell moves from level `k` to level `k + 1` modulo `n` when at least `threshold` of its neighbours hold that
    level, so every level eats its predecessor and random initial levels organize into spiral waves.

    Args:
        label:
            The label of the cell or the lattice group.

        level:
            The initial level, from 0 to `n - 1`.

        n:
            The number of levels.

        threshold:
            The number of neighbours at the following level that advance a cell.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import CyclicCell
        >>> CyclicCell('cell').n
        14.0
    """

    level: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    n: float = parameter(default=14.0, domain=Domain.POSITIVE)
    threshold: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that advances a cell to the following level.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.level, Neighborhood.von_neumann(center=True), _advance_cyclically)]


@register_model(
    field='chemistry/kinetics',
    tags=('lattice', 'excitable media', 'spirals', 'cellular automaton'),
    references=('Griffeath 1988', 'Fisch, Gravner, and Griffeath 1991', 'Dilão 2023'),
    defaults={'until': 200, 'dt': 1},
)
def cyclic_cellular_automaton(
    shape: Sequence[int] = (64, 64),
    *,
    n: int = 14,
    threshold: int = 1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Griffeath's cyclic cellular automaton, whose random levels organize into spiral waves.

    Args:
        shape:
            The number of rows and columns.

        n:
            The number of levels.

        threshold:
            The number of von Neumann neighbours at the following level that advance a cell.

        random_state:
            The seed or generator of the initial levels.

    Returns:
        system:
            The model, with the levels `cells.level`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import cyclic_cellular_automaton
        >>> cyclic_cellular_automaton((10, 10), n=8).elements[0].values['n']
        8
    """
    levels = np.random.default_rng(random_state).integers(0, n, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape))
    return System(components=[CyclicCell.lattice('cells', space, level=levels, n=n, threshold=threshold)])
