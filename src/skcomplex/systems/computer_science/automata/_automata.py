"""Automata systems: elementary rules, life, and turmites, as local rules on cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import BoxNeighborhood, LatticeSpace, MooreNeighborhood, VonNeumannNeighborhood
from ....core.components.systems import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life


class AntCellElement(Element):
    """Cell of Langton's ant: a color, and the heading of the ant on it or -1.

    The local rule of `LangtonsAntSystem` turns the ant right on a white cell and left on a black one, flips the cell's
    color, and moves the ant one cell forward.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        color:
            The initial color, 0 for white and 1 for black.

        ant:
            The initial heading of an ant on the cell, 0 to 3 for north, east, south, and west, or -1 for none.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import AntCellElement
        >>> AntCellElement('cell').ant
        -1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        color: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        ant: Annotated[float, StateSpec(domain=ZZ)] = -1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.color = color
        self.ant = ant


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

    Examples:
        >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> ElementaryAutomatonSystem(90, size=11).prepare().system_.elements[0].prepare().size_
        11
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
    ) -> None:
        super().__init__(label)
        self.rule = rule
        self.size = size
        self.initial = initial

    def _build_system(self) -> System:
        """Return an elementary cellular automaton by its Wolfram number.

        By default it starts from a single live cell.
        """
        values = np.zeros(self.size) if self.initial is None else np.asarray(self.initial, dtype=float)
        if self.initial is None:
            values[self.size // 2] = 1.0
        return System(
            self.label,
            elements=[
                LatticeElement(
                    'cells',
                    element=BinaryCellElement,
                    space=LatticeSpace(self.size),
                    values={'alive': values},
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'rule',
                    target='cells',
                    state='alive',
                    rule=Wolfram(self.rule),
                    neighborhood=BoxNeighborhood(),
                ),
            ],
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

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GameOfLifeSystem
        >>> GameOfLifeSystem((20, 30)).prepare().system_.elements[0].prepare().size_
        600
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
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.density = density
        self.initial = initial
        self.born = born
        self.survive = survive
        self.random_state = random_state

    @staticmethod
    def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
        """Return cells that are 1 with a probability and 0 otherwise."""
        return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)

    def _build_system(self) -> System:
        """Return Conway's Game of Life, or another life-like rule, on a torus."""
        values = (
            self._random_cells(self.shape, self.density, self.random_state)
            if self.initial is None
            else np.asarray(self.initial, dtype=float)
        )
        space = LatticeSpace(tuple(self.shape))
        return System(
            self.label,
            elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': values})],
            interactions=[
                LocalRuleInteraction(
                    'rule',
                    target='cells',
                    state='alive',
                    rule=life(self.born, self.survive),
                    neighborhood=MooreNeighborhood(),
                ),
            ],
        )


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

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LangtonsAntSystem
        >>> int((LangtonsAntSystem((9, 9)).prepare().system_.elements[0].values['ant'] >= 0).sum())
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
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.heading = heading

    @staticmethod
    def _ant_rule(*values: Array) -> tuple[Array, Array]:
        """Return the next colors and ants of Langton's ant.

        They are read from the von Neumann neighbourhoods of both states.
        """
        colors, ants = values[:5], values[5:]
        color = np.where(ants[0] >= 0, 1 - colors[0], colors[0])
        ant = np.full(color.shape, -1.0)
        arrivals = {1: 2, 2: 1, 3: 3, 4: 0}
        for column, heading in arrivals.items():
            turned = np.where(colors[column] == 0, (ants[column] + 1) % 4, (ants[column] + 3) % 4)
            ant = np.where((ants[column] >= 0) & (turned == heading), turned, ant)
        return color, ant

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        ant = np.full(tuple(self.shape), -1.0)
        ant[self.shape[0] // 2, self.shape[1] // 2] = self.heading
        return System(
            self.label,
            elements=[
                LatticeElement(
                    'cells',
                    element=AntCellElement,
                    space=LatticeSpace(tuple(self.shape)),
                    values={'ant': ant},
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'ant',
                    target='cells',
                    state=('color', 'ant'),
                    rule=self._ant_rule,
                    neighborhood=VonNeumannNeighborhood(center=True),
                    reads=('color', 'ant'),
                ),
            ],
        )
