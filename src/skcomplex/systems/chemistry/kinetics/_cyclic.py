"""Cyclic cellular automaton of excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import LatticeSpace, VonNeumannNeighborhood
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class CyclicCellElement(Element):
    """Cell of Griffeath's cyclic cellular automaton, with `n` levels that follow each other in a cycle.

    A cell moves from level `k` to level `k + 1` modulo `n` when at least `threshold` of its neighbours hold that
    level, by the local rule of `CyclicCellularAutomatonSystem`, so every level eats its predecessor and random initial
    levels organize into spiral waves.

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
        >>> from skcomplex.systems.chemistry.kinetics import CyclicCellElement
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


class CyclicCellularAutomatonSystem(BaseSystem):
    """Griffeath's cyclic cellular automaton, whose random levels organize into spiral waves.

    Every cell of a lattice holds one of `n` levels, which follow each other in a cycle, and moves to the following
    level when at least `threshold` of its von Neumann neighbours hold it. The levels are `cells.level`.

    Args:
        shape:
            The number of rows and columns.

        n:
            The number of levels.

        threshold:
            The number of von Neumann neighbours at the following level that advance a cell.

        random_state:
            The seed or generator of the initial levels.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CyclicCellularAutomatonSystem
        >>> CyclicCellularAutomatonSystem((10, 10), n=8).prepare().system_.elements[0].values['n']
        8
    """

    field = 'chemistry/kinetics'
    topics = ('lattice', 'excitable media', 'spirals', 'cellular automaton')
    references = ('Griffeath 1988', 'Fisch, Gravner, and Griffeath 1991', 'Dilão 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        n: int = 14,
        threshold: int = 1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.n = n
        self.threshold = threshold
        self.random_state = random_state

    @staticmethod
    def _advance_cyclically(level: Array, *around: Array, n: float | Array, threshold: float | Array) -> Array:
        """Return the next levels, one step around the cycle where enough neighbours hold the following level."""
        following = np.mod(level + 1, n)
        successors = sum((item == following).astype(float) for item in around)
        return np.where(successors >= threshold, following, level)

    def _build_system(self) -> System:
        """Return the lattice of cells with random levels and the local rule that advances them."""
        shape = tuple(self.shape)
        levels = np.random.default_rng(self.random_state).integers(0, self.n, size=shape).astype(float)
        return System(
            self.label,
            elements=[
                LatticeElement(
                    'cells',
                    element=CyclicCellElement,
                    space=LatticeSpace(shape),
                    values={'level': levels, 'n': self.n, 'threshold': self.threshold},
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'advance',
                    target='cells',
                    state='level',
                    rule=self._advance_cyclically,
                    neighborhood=VonNeumannNeighborhood(center=True),
                ),
            ],
        )
