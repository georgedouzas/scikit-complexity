"""Automata systems: elementary rules, life, turmites, and Turing machines, as local rules on cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import BoxNeighborhood, LatticeSpace, MooreNeighborhood, VonNeumannNeighborhood
from ....core.components.system import System
from ....core.errors import ParameterError
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life
from ._tape_cell import TapeCellElement

BUSY_BEAVER = {
    ('A', 0): (1, 'R', 'B'),
    ('A', 1): (1, 'L', 'C'),
    ('B', 0): (1, 'L', 'A'),
    ('B', 1): (1, 'R', 'B'),
    ('C', 0): (1, 'L', 'B'),
    ('C', 1): (1, 'N', 'H'),
}


HALT = 'H'


MOVES = {'L': -1, 'N': 0, 'R': 1}


def _ant_rule(*values: Array) -> tuple[Array, Array]:
    """Return the next colors and ants of Langton's ant from the von Neumann neighbourhoods of both states."""
    colors, ants = values[:5], values[5:]
    color = np.where(ants[0] >= 0, 1 - colors[0], colors[0])
    ant = np.full(color.shape, -1.0)
    arrivals = {1: 2, 2: 1, 3: 3, 4: 0}
    for column, heading in arrivals.items():
        turned = np.where(colors[column] == 0, (ants[column] + 1) % 4, (ants[column] + 3) % 4)
        ant = np.where((ants[column] >= 0) & (turned == heading), turned, ant)
    return color, ant


class AntCellElement(Element):
    """Cell of Langton's ant: a color, and the heading of the ant on it or -1.

    The local rule of `langtons_ant` turns the ant right on a white cell and left on a black one, flips the cell's
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


def _turing_rule(table: Mapping[tuple[str, int], tuple[int, str, str]]) -> Callable[..., tuple[Array, Array]]:
    """Return the local rule of a Turing machine of an action table, on the symbols and heads of three cells."""
    states = sorted({state for state, _ in table})
    halted = len(states)
    symbols = 1 + max(max(symbol, write) for (_, symbol), (write, _, _) in table.items())
    writes = np.zeros((halted + 1, symbols))
    moves = np.zeros((halted + 1, symbols))
    successors = np.full((halted + 1, symbols), float(halted))
    for (state, symbol), (write, move, successor) in table.items():
        row = states.index(state)
        writes[row, symbol] = write
        moves[row, symbol] = MOVES[move]
        successors[row, symbol] = halted if successor == HALT else states.index(successor)
    for symbol in range(symbols):
        writes[halted, symbol] = symbol

    def rule(*values: Array) -> tuple[Array, Array]:
        symbol, head = values[:3], values[3:]
        read = [cell.astype(int) for cell in symbol]
        state = [np.where(cell >= 0, cell, halted).astype(int) for cell in head]
        written = np.where(head[1] >= 0, writes[state[1], read[1]], symbol[1])
        arrivals = (
            (head[0], moves[state[0], read[0]] == 1, successors[state[0], read[0]]),
            (head[1], moves[state[1], read[1]] == 0, successors[state[1], read[1]]),
            (head[2], moves[state[2], read[2]] == -1, successors[state[2], read[2]]),
        )
        arrived = np.full(written.shape, -1.0)
        for present, toward, successor in arrivals:
            arrived = np.where((present >= 0) & toward, successor, arrived)
        return written, arrived

    return rule


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_system(
    field='computer_science/automata',
    tags=('lattice', 'elementary', 'chaos'),
    references=('Wolfram 1983', 'Charbonneau 2017'),
    defaults={'until': 100, 'dt': 1},
)
def elementary_automaton(rule: int = 30, *, size: int = 101, initial: Sequence[float] | None = None) -> System:
    """Return an elementary cellular automaton by its Wolfram number, started from a single live cell by default.

    Args:
        rule:
            The Wolfram number from 0 to 255.

        size:
            The number of cells of the ring.

        initial:
            The initial cells, or `None` for a single live cell in the middle.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import elementary_automaton
        >>> elementary_automaton(90, size=11).elements[0].prepare().size_
        11
    """
    values = np.zeros(size) if initial is None else np.asarray(initial, dtype=float)
    if initial is None:
        values[size // 2] = 1.0
    return System(
        elements=[
            LatticeElement('cells', element=BinaryCellElement, space=LatticeSpace(size), values={'alive': values}),
        ],
        interactions=[
            LocalRuleInteraction(
                'rule',
                target='cells',
                state='alive',
                rule=Wolfram(rule),
                neighborhood=BoxNeighborhood(),
            ),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('lattice', 'life', 'emergence'),
    references=('Gardner 1970', 'Downey 2018'),
    defaults={'until': 100, 'dt': 1},
)
def game_of_life(
    shape: Sequence[int] = (100, 100),
    *,
    density: float = 0.3,
    initial: npt.ArrayLike | None = None,
    born: Sequence[int] = (3,),
    survive: Sequence[int] = (2, 3),
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Conway's Game of Life, or another life-like rule, on a torus.

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

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import game_of_life
        >>> game_of_life((20, 30)).elements[0].prepare().size_
        600
    """
    values = _random_cells(shape, density, random_state) if initial is None else np.asarray(initial, dtype=float)
    space = LatticeSpace(tuple(shape))
    return System(
        elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': values})],
        interactions=[
            LocalRuleInteraction(
                'rule',
                target='cells',
                state='alive',
                rule=life(born, survive),
                neighborhood=MooreNeighborhood(),
            ),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('lattice', 'ant', 'emergence'),
    references=('Langton 1986',),
    defaults={'until': 11000, 'dt': 100},
)
def langtons_ant(shape: Sequence[int] = (80, 80), *, heading: int = 0) -> System:
    """Return Langton's ant, which wanders chaotically for about 10,000 steps before it builds a periodic highway.

    Args:
        shape:
            The number of rows and columns of the torus.

        heading:
            The initial heading of the ant, 0 to 3 for north, east, south, and west.

    Returns:
        system:
            The system, with the colors `cells.color` and the ant `cells.ant`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import langtons_ant
        >>> int((langtons_ant((9, 9)).elements[0].values['ant'] >= 0).sum())
        1
    """
    ant = np.full(tuple(shape), -1.0)
    ant[shape[0] // 2, shape[1] // 2] = heading
    return System(
        elements=[
            LatticeElement('cells', element=AntCellElement, space=LatticeSpace(tuple(shape)), values={'ant': ant}),
        ],
        interactions=[
            LocalRuleInteraction(
                'ant',
                target='cells',
                state=('color', 'ant'),
                rule=_ant_rule,
                neighborhood=VonNeumannNeighborhood(center=True),
                reads=('color', 'ant'),
            ),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('lattice', 'turing machine', 'busy beaver', 'computation'),
    references=('Turing 1936', 'Radó 1962', 'Downey 2018'),
    defaults={'until': 20, 'dt': 1},
)
def turing_machine(
    size: int = 21,
    *,
    table: Mapping[tuple[str, int], tuple[int, str, str]] | None = None,
    start: str = 'A',
) -> System:
    """Return a Turing machine, a head with an internal state that reads, writes, and moves on a tape of symbols.

    The action table maps the internal state of the head and the symbol under it to the symbol it writes, its move
    `'L'`, `'R'`, or `'N'`, and its next internal state, `'H'` to halt. Every step is a local rule on the cells of a
    ring of tape: the cell under the head takes the written symbol, and a cell takes the head that moves onto it. The
    default table is the busy beaver of three states and two symbols, which writes the most ones a halting machine of
    its size can write on a blank tape: it halts after 13 steps with six ones on the tape.

    Args:
        size:
            The number of cells of the ring of tape, large enough that the head never wraps around.

        table:
            The action table, or `None` for the three-state busy beaver.

        start:
            The internal state of the head at the start, on the middle cell of a blank tape.

    Returns:
        system:
            The system, with the symbols `tape.symbol` and the head `tape.head`, which holds the index of the internal
            state of the head, in the sorted order of the states of the table, on its cell, that index plus one when
            the head has halted, and -1 elsewhere.

    Raises:
        ParameterError:
            The start is not a state of the table, or a move is not `'L'`, `'R'`, or `'N'`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import turing_machine
        >>> int(turing_machine(11).elements[0].values['head'][5])
        0
    """
    actions = BUSY_BEAVER if table is None else table
    states = sorted({state for state, _ in actions})
    if start not in states or any(move not in MOVES for _, move, _ in actions.values()):
        error_msg = f'The start {start!r} needs to be a state of the table, and every move one of L, R, or N.'
        raise ParameterError(error_msg)
    head = np.full(size, -1.0)
    head[size // 2] = states.index(start)
    tape = LatticeElement('tape', element=TapeCellElement, space=LatticeSpace(size), values={'head': head})
    return System(
        elements=[tape],
        interactions=[
            LocalRuleInteraction(
                'machine',
                target='tape',
                state=('symbol', 'head'),
                rule=_turing_rule(actions),
                neighborhood=BoxNeighborhood(),
                reads=('symbol', 'head'),
            ),
        ],
    )
