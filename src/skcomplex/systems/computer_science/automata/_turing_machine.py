"""Turing machine: a head that reads, writes, and moves on a tape, as a local rule.

The head has an internal state.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import LocalRuleLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ....core.types import Array

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
THREE_CELLS = LatticeNeighborhood(((-1,), (0,), (1,)))


class TapeCellElement(Element):
    """Cell of the tape of a Turing machine.

    It holds a symbol, and the internal state of the head on it or -1. A `TuringMachineInteraction` lets the head on a
    cell write the symbol of its action table, move one cell left or right, or stay, and take its next internal state.
    A halted head keeps its cell and the tape stays unchanged. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        symbol:
            The initial symbol, or one symbol per cell.

        head:
            The internal state of the head on the cell, from 0, or -1 for no head, or one value per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TapeCellElement
        >>> TapeCellElement('cell').head
        -1
    """

    def __init__(
        self,
        label: str,
        *,
        symbol: Annotated[float | Array, NN] = 0,
        head: Annotated[float | Array, ZZ] = -1,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.symbol = symbol
        self.head = head


class TuringMachineInteraction(Interaction):
    """Head of a Turing machine that reads, writes, and moves on the cells of a tape.

    The action table maps the internal state of the head and the symbol under it to the symbol it writes, its move
    `'L'`, `'R'`, or `'N'`, and its next internal state, `'H'` to halt. Every step is a local rule on three cells: the
    cell under the head takes the written symbol, and a cell takes the head that moves onto it.

    Args:
        label:
            The label of the interaction.

        target:
            The cells of the tape, whose states `symbol` and `head` the machine changes.

        table:
            The action table, or `None` for the three-state busy beaver.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TuringMachineInteraction
        >>> TuringMachineInteraction('machine', target='tape').table is None
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'tape',
        table: Mapping[tuple[str, int], tuple[int, str, str]] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.table = table

    def _get_function(self) -> Callable[..., tuple[Array, Array]]:
        """Return the local rule of the action table, which reads the symbols and heads of three cells."""
        table = BUSY_BEAVER if self.table is None else self.table
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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the step of the machine on three cells as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            states = (context.target.symbol, context.target.head)
            self.laws_: list[BaseComponent] = [
                LocalRuleLaw(states, THREE_CELLS, self._get_function(), reads=states, label=self.label),
            ]


class TuringMachineSystem(BaseSystem):
    """A Turing machine, a head that reads, writes, and moves on a tape of symbols.

    The head has an internal state.

    The action table maps the internal state of the head and the symbol under it to the symbol it writes, its move
    `'L'`, `'R'`, or `'N'`, and its next internal state, `'H'` to halt. Every step is a local rule on the cells of a
    ring of tape: the cell under the head takes the written symbol, and a cell takes the head that moves onto it. The
    default table is the busy beaver of three states and two symbols, which writes the most ones a halting machine of
    its size can write on a blank tape: it halts after 13 steps with six ones on the tape. The system has the symbols
    `tape.symbol` and the head `tape.head`, which holds the index of the internal state of the head, in the sorted order
    of the states of the table, on its cell, that index plus one when the head has halted, and -1 elsewhere.

    Args:
        size:
            The number of cells of the ring of tape, large enough that the head never wraps around.

        table:
            The action table, or `None` for the three-state busy beaver.

        start:
            The internal state of the head at the start, on the middle cell of a blank tape.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The start is not a state of the table, or a move is not `'L'`, `'R'`, or `'N'`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TuringMachineSystem
        >>> int(TuringMachineSystem(11).prepare().system_.elements[0].head[5])
        0
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'turing machine', 'busy beaver', 'computation')
    references = ('Turing 1936', 'Radó 1962', 'Downey 2018')
    until = 20
    dt = 1

    def __init__(
        self,
        size: int = 21,
        *,
        table: Mapping[tuple[str, int], tuple[int, str, str]] | None = None,
        start: str = 'A',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.table = table
        self.start = start

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a blank tape with the head on its middle cell and the machine."""
        super()._prepare(context)
        actions = BUSY_BEAVER if self.table is None else self.table
        states = sorted({state for state, _ in actions})
        if self.start not in states or any(move not in MOVES for _, move, _ in actions.values()):
            error_msg = f'The start {self.start!r} needs to be a state of the table, and every move one of L, R, or N.'
            raise ParameterError(error_msg)
        head = np.full(self.size, -1.0)
        head[self.size // 2] = states.index(self.start)
        self.system_ = System(
            self.label,
            elements=[TapeCellElement('tape', symbol=np.zeros(self.size), head=head, space=LatticeSpace(self.size))],
            interactions=[TuringMachineInteraction('machine', target='tape', table=self.table)],
        )
