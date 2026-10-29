"""Cyclic cellular automaton of excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.laws import BaseLaw, LocalRuleLaw
from ....core.spaces import LatticeSpace, VonNeumannNeighborhood
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model


def _advance_cyclically(level: Array, *around: Array, n: float | Array, threshold: float | Array) -> Array:
    """Return the next levels, one step around the cycle where enough neighbours hold the following level."""
    following = np.mod(level + 1, n)
    successors = sum((item == following).astype(float) for item in around)
    return np.where(successors >= threshold, following, level)


class CyclicCellElement(Element):
    """Cell of Griffeath's cyclic cellular automaton, with `n` levels that follow each other in a cycle.

    A cell moves from level `k` to level `k + 1` modulo `n` when at least `threshold` of its neighbours hold that
    level, so every level eats its predecessor and random initial levels organize into spiral waves.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial level, from 0 to `n - 1`.

        n:
            The number of levels.

        threshold:
            The number of neighbours at the following level that advance a cell.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import CyclicCellElement
        >>> CyclicCellElement('cell').n
        14.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 14.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.n = n
        self.threshold = threshold

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that advances a cell to the following level.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRuleLaw(s.level, VonNeumannNeighborhood(center=True), _advance_cyclically)]


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
    return System(
        elements=[
            LatticeElement(
                'cells',
                element=CyclicCellElement,
                space=space,
                values={'level': levels, 'n': n, 'threshold': threshold},
            ),
        ],
    )
