"""Automata systems: elementary rules and life as local rules on cells, and turmites."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import LatticeNeighborhood, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import AnyArray, Array, Role
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram
from ._life_rule import LifeRuleInteraction

ELEMENTARY = LatticeNeighborhood(((-1,), (0,), (1,)))
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


class AntView(AgentViewProtocol, Protocol):
    """View of Langton's ant, with its states as attributes."""

    row: int
    column: int
    heading: int
    colors: npt.NDArray[np.uint8] | frozenset[tuple[int, int]]


class AntElement(Element):
    """Langton's ant: its cell, its heading, and the colours of the cells it walks on.

    A `LangtonsAntInteraction` turns the ant right on a white cell and left on a black one, flips the colour of that
    cell, and moves the ant one cell forward. The element is one agent, so a run records its states at every frame.

    Args:
        label:
            The label of the ant.

        row:
            The row of the cell of the ant.

        column:
            The column of the cell of the ant.

        heading:
            The heading of the ant, 0 to 3 for north, east, south, and west.

        colors:
            An object array that holds the colours of the cells, either one integer array with 0 for white and 1 for
            black on a torus or one frozenset of the row and column of every black cell on the plane, or `None` for
            none.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import AntElement
        >>> AntElement('ant', row=4, column=4, heading=1).heading
        1
    """

    def __init__(
        self,
        label: str,
        *,
        row: Annotated[int, ZZ] = 0,
        column: Annotated[int, ZZ] = 0,
        heading: Annotated[int, {0, 1, 2, 3}] = 0,
        colors: AnyArray | None = None,
    ) -> None:
        super().__init__(label, size=1)
        self.row = row
        self.column = column
        self.heading = heading
        self.colors = colors


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

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the local rule of the cells as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        alive = context.target.alive
        self.laws_: list[BaseLaw] = [
            UpdateLaw(alive, Wolfram(self.rule), neighborhood=ELEMENTARY, label=self.label),
        ]


class LangtonsAntInteraction(Interaction):
    """Move of Langton's ant over the coloured cells of a torus or of the plane.

    The ant turns right on a white cell and left on a black one, flips the colour of that cell, and moves one cell
    forward. On a torus, whose colours are an array, it crosses the edges. On the plane, whose colours are the set of
    black cells, it never wraps. A step reads and changes the cell of the ant alone.

    Args:
        label:
            The label of the interaction.

        target:
            The ant whose states `row`, `column`, `heading`, and `colors` the move changes.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LangtonsAntInteraction
        >>> LangtonsAntInteraction('move', target='ant').target
        'ant'
    """

    def __init__(self, label: str, *, target: Role = 'ant') -> None:
        super().__init__(label)
        self.target = target

    def _move(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Turn the ant on its cell, flip the colour of the cell, and step forward.

        The colours are copied before the flip, so the colours a run recorded at earlier frames stay as they were.
        """
        del system
        ant = cast('AntView', agent)
        if isinstance(ant.colors, frozenset):
            self._move_on_plane(ant, ant.colors)
        else:
            self._move_on_torus(ant, ant.colors.copy())

    @staticmethod
    def _move_on_plane(ant: AntView, black: frozenset[tuple[int, int]]) -> None:
        """Turn the ant on its cell of the plane, flip the cell, and step forward."""
        cell = (ant.row, ant.column)
        heading = (ant.heading + (3 if cell in black else 1)) % 4
        ant.row += MOVES[heading][0]
        ant.column += MOVES[heading][1]
        ant.heading = heading
        ant.colors = black ^ {cell}

    @staticmethod
    def _move_on_torus(ant: AntView, colors: npt.NDArray[np.uint8]) -> None:
        """Turn the ant on its cell of the torus, flip the cell, and step across edges."""
        cell = (ant.row, ant.column)
        heading = (ant.heading + (3 if colors[cell] else 1)) % 4
        colors[cell] = 1 - colors[cell]
        rows, columns = colors.shape
        ant.row = (ant.row + MOVES[heading][0]) % rows
        ant.column = (ant.column + MOVES[heading][1]) % columns
        ant.heading = heading
        ant.colors = colors

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the move of the ant as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._move, label=self.label)]


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

    From a blank start it wanders chaotically for about 10,000 steps before it builds the highway, a cycle of 104 steps
    that moves it two cells along a diagonal. The system has the ant `ant`, with its cell `ant.row` and `ant.column`,
    its heading `ant.heading`, and the colours `ant.colors`, which a run records at every frame. On a torus the frames
    hold arrays of the colours. On the unbounded plane, which `shape=None` gives, the ant never wraps and the frames
    hold frozensets of the row and column of every black cell, so memory grows with the black cells and not with a
    lattice. A step changes the cell of the ant alone, so long runs stay fast.

    Args:
        shape:
            The number of rows and columns of the torus, or `None` for the unbounded plane.

        colors:
            The initial colours of the cells, or `None` for a blank start. On a torus they are an array of the shape
            with 0 for white and 1 for black. On the plane they are a sequence of the row and column of every black
            cell.

        start:
            The row and column of the initial cell of the ant, or `None` for the middle of the torus or the cell
            `(0, 0)` of the plane.

        heading:
            The initial heading of the ant, 0 to 3 for north, east, south, and west.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The colours are not 0 or 1 with the shape of the torus, the black cells of the plane are not pairs of
            integers, or the start lies outside the torus.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LangtonsAntSystem
        >>> trajectory = LangtonsAntSystem((9, 9)).prepare().simulate(until=2, dt=1)
        >>> trajectory.get_values('ant.column').ravel().tolist()
        [4, 5, 5]
        >>> plane = LangtonsAntSystem(None).prepare().simulate(until=2, dt=1)
        >>> sorted(plane.get_values('ant.colors')[-1, 0])
        [(0, 0), (0, 1)]
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'ant', 'emergence')
    references = ('Langton 1986',)
    until = 11000
    dt = 100

    def __init__(
        self,
        shape: Sequence[int] | None = (80, 80),
        *,
        colors: npt.ArrayLike | None = None,
        start: Sequence[int] | None = None,
        heading: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.colors = colors
        self.start = start
        self.heading = heading

    def _get_colors(self) -> AnyArray:
        """Return an object array that holds the initial colours of the torus or plane.

        Returns:
            colors:
                One array of colours on a torus, or one set of black cells on the plane.

        Raises:
            ValidationError:
                The colours are not 0 or 1 with the shape of the torus, or the black cells are not pairs of integers.
        """
        holder = np.empty(1, dtype=object)
        if self.shape is None:
            holder[0] = self._get_black_cells()
            return holder
        shape = tuple(self.shape)
        colors = np.zeros(shape) if self.colors is None else np.asarray(self.colors)
        if colors.shape != shape or not np.isin(colors, (0, 1)).all():
            error_msg = (
                f'The colors must be 0 or 1 with the shape {shape} of the torus, but they have the shape '
                f'{colors.shape} and the values {np.unique(colors).tolist()}.'
            )
            raise ValidationError(error_msg)
        holder[0] = colors.astype(np.uint8)
        return holder

    def _get_black_cells(self) -> frozenset[tuple[int, int]]:
        """Return the row and column of every initial black cell of the plane.

        Returns:
            cells:
                The black cells, as pairs of a row and a column.

        Raises:
            ValidationError:
                The black cells are not pairs of integers.
        """
        cells = np.empty((0, 2), dtype=int) if self.colors is None else np.asarray(self.colors)
        if cells.size == 0:
            return frozenset()
        if cells.shape != (len(cells), 2) or not np.issubdtype(cells.dtype, np.integer):
            error_msg = (
                'On the plane the colors must be the row and column of every black cell, pairs of integers, but they '
                f'have the shape {cells.shape} and the type {cells.dtype}.'
            )
            raise ValidationError(error_msg)
        return frozenset((row, column) for row, column in cells.tolist())

    def _get_start(self) -> tuple[int, int]:
        """Return the row and column of the initial cell of the ant.

        Returns:
            start:
                The row and the column of the ant at the start.

        Raises:
            ValidationError:
                The start lies outside the torus.
        """
        if self.shape is None:
            row, column = (0, 0) if self.start is None else self.start
            return int(row), int(column)
        rows, columns = self.shape
        row, column = (rows // 2, columns // 2) if self.start is None else self.start
        if not (0 <= row < rows and 0 <= column < columns):
            error_msg = f'The start {(row, column)} lies outside the torus of the shape {(rows, columns)}.'
            raise ValidationError(error_msg)
        return int(row), int(column)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the ant on the coloured cells of a torus or of the plane."""
        super()._prepare(context)
        row, column = self._get_start()
        ant = AntElement('ant', row=row, column=column, heading=int(self.heading), colors=self._get_colors())
        self.system_ = System(
            self.label,
            elements=[ant],
            interactions=[LangtonsAntInteraction('move', target='ant')],
        )
