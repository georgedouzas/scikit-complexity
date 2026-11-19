"""Automata systems: elementary rules, life, and turmites, as local rules on cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import (
    BaseSpace,
    LatticeNeighborhood,
    LatticeSpace,
    RadiusNeighborhood,
)
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram
from ._life_rule import LifeRuleInteraction

ELEMENTARY = LatticeNeighborhood(((-1,), (0,), (1,)))
VON_NEUMANN = RadiusNeighborhood(1, metric='manhattan', center=True)


class AntCellElement(Element):
    """Cell of Langton's ant: a color, and the heading of the ant on it or -1.

    A `LangtonsAntInteraction` turns the ant right on a white cell and left on a black one, flips the cell's color, and
    moves the ant one cell forward. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        color:
            The initial color, 0 for white and 1 for black, or one color per cell.

        ant:
            The initial heading of an ant on the cell, 0 to 3 for north, east, south, and west, or -1 for none, or one
            heading per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import AntCellElement
        >>> AntCellElement('cell').ant
        -1
    """

    def __init__(
        self,
        label: str,
        *,
        color: Annotated[float | Array, {0, 1}] = 0,
        ant: Annotated[float | Array, ZZ] = -1,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.color = color
        self.ant = ant


class ElementaryRuleInteraction(Interaction):
    """Elementary rule of a line of cells, given by its Wolfram number.

    Every cell reads itself and its two neighbours, which make one of eight patterns, and the bit of the Wolfram number
    at the place of that pattern gives its next value. Rule 30 looks random, and rule 90 draws Pascal's triangle
    modulo 2.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        rule:
            The Wolfram number from 0 to 255.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import ElementaryRuleInteraction
        >>> ElementaryRuleInteraction('rule', target='cells', rule=90).rule
        90
    """

    def __init__(self, label: str, *, target: Role = 'cells', rule: int = 30) -> None:
        super().__init__(label)
        self.target = target
        self.rule = rule

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the local rule of the cells as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            alive = context.target.alive
            self.laws_: list[BaseComponent] = [
                UpdateLaw(alive, Wolfram(self.rule), neighborhood=ELEMENTARY, label=self.label),
            ]


class LangtonsAntInteraction(Interaction):
    """Move of Langton's ant over the colored cells.

    The ant turns right on a white cell and left on a black one, flips the color of the cell it leaves, and moves one
    cell forward. Every cell holds its color and the heading of an ant on it, or -1 for none.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose states `color` and `ant` the ant changes.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LangtonsAntInteraction
        >>> LangtonsAntInteraction('ant', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _get_next(*values: Array) -> tuple[Array, Array]:
        """Return the next colors and ants from the von Neumann neighbourhoods of both states."""
        colors, ants = values[:5], values[5:]
        color = np.where(ants[0] >= 0, 1 - colors[0], colors[0])
        ant = np.full(color.shape, -1.0)
        arrivals = {1: 2, 2: 1, 3: 3, 4: 0}
        for column, heading in arrivals.items():
            turned = np.where(colors[column] == 0, (ants[column] + 1) % 4, (ants[column] + 3) % 4)
            ant = np.where((ants[column] >= 0) & (turned == heading), turned, ant)
        return color, ant

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the move of the ant as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            states = (context.target.color, context.target.ant)
            self.laws_: list[BaseComponent] = [
                UpdateLaw(states, self._get_next, neighborhood=VON_NEUMANN, label=self.label),
            ]


class ElementaryAutomatonSystem(BaseSystem):
    """An elementary cellular automaton by its Wolfram number.

    By default it starts from a single live cell. The system has the cells `cells.alive`.

    Args:
        rule:
            The Wolfram number from 0 to 255.

        size:
            The number of cells of the ring.

        initial:
            The initial cells, or `None` for a single live cell in the middle.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> trajectory = ElementaryAutomatonSystem(90, size=7).prepare().simulate(until=2)
        >>> trajectory.get_values('cells.alive')[-1].astype(int).tolist()
        [0, 1, 0, 0, 0, 1, 0]
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'elementary', 'chaos')
    references = ('Wolfram 1983', 'Charbonneau 2017')
    until = 100
    dt = 1

    def __init__(
        self,
        rule: int = 30,
        *,
        size: int = 101,
        initial: Sequence[float] | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.rule = rule
        self.size = size
        self.initial = initial

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of cells and its elementary rule."""
        super()._prepare(context)
        values = np.zeros(self.size) if self.initial is None else np.asarray(self.initial, dtype=float)
        if self.initial is None:
            values[self.size // 2] = 1.0
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=values, space=LatticeSpace(self.size))],
            interactions=[ElementaryRuleInteraction('rule', target='cells', rule=self.rule)],
        )


class GameOfLifeSystem(BaseSystem):
    """Conway's Game of Life, or another life-like rule, on a torus.

    The system has the cells `cells.alive`.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a cell starts alive, when no initial cells are given.

        initial:
            The initial cells, an array of the shape, or `None` for random cells.

        born:
            The numbers of live neighbours that bring a dead cell to life.

        survive:
            The numbers of live neighbours that keep a live cell alive.

        random_state:
            The seed or generator of the random initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GameOfLifeSystem
        >>> GameOfLifeSystem((20, 30)).prepare().simulate(until=1).get_lattice('cells.alive').shape
        (2, 20, 30)
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'life', 'emergence')
    references = ('Gardner 1970', 'Downey 2018')
    until = 100
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        density: float = 0.3,
        initial: npt.ArrayLike | None = None,
        born: Sequence[int] = (3,),
        survive: Sequence[int] = (2, 3),
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.density = density
        self.initial = initial
        self.born = born
        self.survive = survive
        self.random_state = random_state

    def _get_cells(self) -> Array:
        """Return the initial cells, given or alive with the probability `density`."""
        if self.initial is not None:
            return np.asarray(self.initial, dtype=float)
        random = np.random.default_rng(self.random_state).random(tuple(self.shape))
        return (random < self.density).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells of a torus and their life-like rule."""
        super()._prepare(context)
        cells = BinaryCellElement('cells', alive=self._get_cells().ravel(), space=LatticeSpace(tuple(self.shape)))
        rule = LifeRuleInteraction('rule', target='cells', born=self.born, survive=self.survive)
        self.system_ = System(self.label, elements=[cells], interactions=[rule])


class LangtonsAntSystem(BaseSystem):
    """Langton's ant, which builds a periodic highway after a chaotic transient.

    It wanders chaotically for about 10,000 steps before it builds the highway. The system has the colors `cells.color`
    and the ant `cells.ant`.

    Args:
        shape:
            The number of rows and columns of the torus.

        heading:
            The initial heading of the ant, 0 to 3 for north, east, south, and west.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LangtonsAntSystem
        >>> int((LangtonsAntSystem((9, 9)).prepare().system_.elements[0].ant >= 0).sum())
        1
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'ant', 'emergence')
    references = ('Langton 1986',)
    until = 11000
    dt = 100

    def __init__(
        self,
        shape: Sequence[int] = (80, 80),
        *,
        heading: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.heading = heading

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the colored cells and the ant on them."""
        super()._prepare(context)
        ant = np.full(tuple(self.shape), -1.0)
        ant[self.shape[0] // 2, self.shape[1] // 2] = self.heading
        cells = AntCellElement(
            'cells',
            color=np.zeros(ant.size),
            ant=ant.ravel(),
            space=LatticeSpace(tuple(self.shape)),
        )
        self.system_ = System(
            self.label,
            elements=[cells],
            interactions=[LangtonsAntInteraction('ant', target='cells')],
        )
