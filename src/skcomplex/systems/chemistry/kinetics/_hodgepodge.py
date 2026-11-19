"""Chemical kinetics systems: reactions and excitable media."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

MOORE = RadiusNeighborhood(1, metric='chebyshev', center=True)


class HodgepodgeCellElement(Element):
    """Cell of the hodgepodge machine of Gerhardt and Schuster.

    It is healthy at 0, infected below `n`, and ill at `n`.

    Healthy cells catch the infection from their infected and ill neighbours, infected cells worsen by the mean of their
    neighbourhood plus `g`, and ill cells heal, by the local rule of `HodgepodgeSystem`.

    Args:
        label:
            The label of the cell or the lattice group.

        level:
            The initial level of infection from 0 to `n`.

        n:
            The level of illness.

        k1:
            The weight of infected neighbours for healthy cells.

        k2:
            The weight of ill neighbours for healthy cells.

        g:
            The speed of the infection.

        space:
            The lattice of the cells, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import HodgepodgeCellElement
        >>> HodgepodgeCellElement('cell').n
        100.0
    """

    def __init__(
        self,
        label: str,
        *,
        level: Annotated[object, NN] = 0,
        n: Annotated[float, RealSet.open(0, oo)] = 100.0,
        k1: Annotated[float, RealSet.open(0, oo)] = 2.0,
        k2: Annotated[float, RealSet.open(0, oo)] = 3.0,
        g: Annotated[float, RealSet.closed_open(0, oo)] = 34.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.level = level
        self.n = n
        self.k1 = k1
        self.k2 = k2
        self.g = g


class HodgepodgeInteraction(Interaction):
    """Infection, worsening, and healing of the cells of the hodgepodge machine.

    A healthy cell catches the infection from its infected and ill neighbours, an infected cell worsens by the mean of
    its Moore neighbourhood plus `g`, and an ill cell heals, with `n`, `k1`, `k2`, and `g` the parameters of its cell.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `level` the rule updates.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import HodgepodgeInteraction
        >>> HodgepodgeInteraction('infection', target='cells').target
        'cells'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _infect(
        level: Array,
        *around: Array,
        n: float | Array,
        k1: float | Array,
        k2: float | Array,
        g: float | Array,
    ) -> Array:
        """Return the next levels.

        Healthy cells catch the infection, infected cells worsen, and ill cells heal.
        """
        neighbours = np.stack(around)
        infected = ((neighbours > 0) & (neighbours < n)).sum(axis=0)
        ill = (neighbours >= n).sum(axis=0)
        healthy = np.floor(infected / k1) + np.floor(ill / k2)
        worsening = np.minimum(np.floor((level + neighbours.sum(axis=0)) / (infected + ill + 1)) + g, n)
        return np.where(level <= 0, np.minimum(healthy, n), np.where(level >= n, 0.0, worsening))

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the infection of the cells as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            level = context.target.level
            self.laws_: list[BaseComponent] = [UpdateLaw(level, self._infect, neighborhood=MOORE, label=self.label)]


class HodgepodgeSystem(BaseSystem):
    """The hodgepodge machine, a chemical reaction whose waves form spirals.

    Every cell of a lattice holds a level of infection from 0 to `n`. Healthy cells catch the infection from their
    infected and ill Moore neighbours, infected cells worsen by the mean of their neighbourhood plus `g`, and ill cells
    heal. The levels are `cells.level`.

    Args:
        shape:
            The number of rows and columns.

        n:
            The level of illness.

        k1:
            The weight of infected neighbours for healthy cells.

        k2:
            The weight of ill neighbours for healthy cells.

        g:
            The speed of the infection.

        random_state:
            The seed or generator of the initial levels.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import HodgepodgeSystem
        >>> HodgepodgeSystem((10, 10)).prepare().system_.elements[0].g
        34.0
    """

    field = 'chemistry/kinetics'
    topics = ('lattice', 'excitable media', 'spirals')
    references = ('Gerhardt and Schuster 1989', 'Charbonneau 2017')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        n: float = 100.0,
        k1: float = 2.0,
        k2: float = 3.0,
        g: float = 34.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.n = n
        self.k1 = k1
        self.k2 = k2
        self.g = g
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the lattice of cells with random levels and the rule of the infection."""
        super()._prepare(context)
        shape = tuple(self.shape)
        levels = np.random.default_rng(self.random_state).integers(0, int(self.n) + 1, size=shape).astype(float)
        self.system_ = System(
            self.label,
            elements=[
                HodgepodgeCellElement(
                    'cells',
                    level=levels,
                    n=self.n,
                    k1=self.k1,
                    k2=self.k2,
                    g=self.g,
                    space=LatticeSpace(shape),
                ),
            ],
            interactions=[
                HodgepodgeInteraction('infection', target='cells'),
            ],
        )
