"""Cyclic cellular automaton of excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

VON_NEUMANN = RadiusNeighborhood(1, metric='manhattan', center=True)


class CyclicCellElement(Element):
    """Cell of Griffeath's cyclic cellular automaton, with `n` levels.

    The levels follow each other in a cycle.

    A cell moves from level `k` to level `k + 1` modulo `n` when at least `threshold` of its neighbours hold that
    level, by the local rule of `CyclicCellularAutomatonSystem`, so every level eats its predecessor and random initial
    levels organize into spiral waves.

    Args:
        label:
            The label of the cell or the lattice group.

        level:
            The initial level, from 0 to `n - 1`.

        n:
            The number of levels.

        threshold:
            The number of neighbours at the following level that advance a cell.

        space:
            The lattice of the cells, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CyclicCellElement
        >>> CyclicCellElement('cell').n
        14.0
    """

    def __init__(
        self,
        label: str,
        *,
        level: Annotated[object, NN] = 0,
        n: Annotated[float, RealSet.open(0, oo)] = 14.0,
        threshold: Annotated[float, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.level = level
        self.n = n
        self.threshold = threshold


class CyclicAdvanceInteraction(Interaction):
    """Advance of the cells of Griffeath's cyclic automaton around their cycle of levels.

    A cell moves from level `k` to level `k + 1` modulo `n` when at least `threshold` of its von Neumann neighbours
    hold that level, with `n` and `threshold` the parameters of its cell.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `level` the rule updates.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CyclicAdvanceInteraction
        >>> CyclicAdvanceInteraction('advance', target='cells').target
        'cells'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _advance_cyclically(
        level: Array,
        *around: Array,
        n: float | Array,
        threshold: float | Array,
        **kwargs: object,
    ) -> Array:
        """Return the next levels, one step around the cycle.

        A cell steps where enough neighbours hold the following level.
        """
        following = np.mod(level + 1, n)
        successors = sum((item == following).astype(float) for item in around)
        return np.where(successors >= threshold, following, level)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the advance of the levels as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        level = context.target.level
        self.laws_: list[BaseLaw] = [
            UpdateLaw(level, self._advance_cyclically, neighborhood=VON_NEUMANN, label=self.label),
        ]


class CyclicCellularAutomatonSystem(BaseSystem):
    """Griffeath's cyclic cellular automaton.

    Its random levels organize into spiral waves.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CyclicCellularAutomatonSystem
        >>> CyclicCellularAutomatonSystem((10, 10), n=8).prepare().system_.elements[0].n
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.n = n
        self.threshold = threshold
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the lattice of cells with random levels and the rule that advances them."""
        super()._prepare(context)
        shape = tuple(self.shape)
        levels = np.random.default_rng(self.random_state).integers(0, self.n, size=shape).astype(float)
        self.system_ = System(
            self.label,
            elements=[
                CyclicCellElement('cells', level=levels, n=self.n, threshold=self.threshold, space=LatticeSpace(shape)),
            ],
            interactions=[
                CyclicAdvanceInteraction('advance', target='cells'),
            ],
        )
