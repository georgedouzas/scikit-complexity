"""Turing machine: a head that reads, writes, and moves on a tape, as a local rule.

The head has an internal state.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import BallNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ParameterError
from ....core.specs import StateSpec, VisualSpec
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


class TapeCellElement(Element):
    """Cell of the tape of a Turing machine.

    It holds a symbol, and the internal state of the head on it or -1. The local rule of `TuringMachineSystem` lets the
    head on a cell write the symbol of its action table, move one cell left or right, or stay, and take its next
    internal state. A halted head keeps its cell and the tape stays unchanged.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol.

        head:
            The internal state of the head on the cell, from 0, or -1 for no head.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TapeCellElement
        >>> TapeCellElement('cell').head
        -1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
        head: Annotated[float, StateSpec(domain=ZZ)] = -1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol
        self.head = head


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

    Raises:
        ParameterError:
            The start is not a state of the table, or a move is not `'L'`, `'R'`, or `'N'`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TuringMachineSystem
        >>> int(TuringMachineSystem(11).prepare().system_.elements[0].values['head'][5])
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
    ) -> None:
        super().__init__(label)
        self.size = size
        self.table = table
        self.start = start

    @staticmethod
    def _get_turing_rule(table: Mapping[tuple[str, int], tuple[int, str, str]]) -> Callable[..., tuple[Array, Array]]:
        """Return the local rule of a Turing machine of an action table.

        The rule reads the symbols and heads of three cells.
        """
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

    def _build_system(self) -> CompositeSystem:
        """Return a Turing machine, a head that reads, writes, and moves on a tape.

        The tape holds symbols. The head has an internal state.
        """
        actions = BUSY_BEAVER if self.table is None else self.table
        states = sorted({state for state, _ in actions})
        if self.start not in states or any(move not in MOVES for _, move, _ in actions.values()):
            error_msg = f'The start {self.start!r} needs to be a state of the table, and every move one of L, R, or N.'
            raise ParameterError(error_msg)
        head = np.full(self.size, -1.0)
        head[self.size // 2] = states.index(self.start)
        tape = GroupElement(
            'tape',
            element=TapeCellElement,
            space=LatticeSpace(self.size),
            values={'head': head},
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[tape],
            interactions=[
                LocalRuleInteraction(
                    'machine',
                    target='tape',
                    state=('symbol', 'head'),
                    rule=self._get_turing_rule(actions),
                    neighborhood=BallNeighborhood(n_dim=1, center='inline'),
                    reads=('symbol', 'head'),
                ),
            ],
        )
