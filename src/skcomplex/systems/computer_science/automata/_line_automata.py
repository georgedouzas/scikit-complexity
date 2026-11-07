"""Binary automata on the integers with long memory sets: a nilpotent automaton and an involution."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.components.elements import LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import LatticeNeighborhood, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec
from ....core.types import Array
from ._binary_cell import BinaryCellElement

MIN_NILPOTENT_CLASS = 2
MIN_INVOLUTION_RUN = 3


def _random_cells(size: int, density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < density).astype(float)


def _line_system(label: str, rule: object, offsets: range, values: Array) -> System:
    """Return a binary automaton on a ring that reads a window of offsets."""
    return System(
        label,
        elements=[
            LatticeElement(
                'cells',
                element=BinaryCellElement,
                space=LatticeSpace(values.size),
                values={'alive': values},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'rule',
                target='cells',
                state='alive',
                rule=rule,
                neighborhood=LatticeNeighborhood(tuple((offset,) for offset in offsets)),
            ),
        ],
    )


class NilpotentAutomatonSystem(BaseSystem):
    """A binary automaton on a ring that is nilpotent of class `n`: `n` steps send every configuration to 0.

    The memory set is `{-1, 0, ..., n - 1}`, and a cell becomes 0 when the whole window holds ones or when it is the
    left end of a run of ones, `p(-1) = 0` and `p(0) = 1`, and keeps its value otherwise (Ceccherini-Silberstein and
    Coornaert 2023, Exercise 1.15). After one step every run of ones is shorter than `n`, and every further step removes
    the left end of each run, so `n` steps reach 0 and the run of `n + 1` ones needs all of them. The exercise prints
    the memory set `{-1, ..., n - 2}`, whose class is `n - 1`, while its solution relies on `{-1, ..., n - 1}`, which
    this system takes. The system has the cells `cells.alive`.

    Args:
        n:
            The nilpotency class, at least 2.

        size:
            The number of cells of the ring.

        density:
            The probability that a cell starts at 1, when no initial cells are given.

        initial:
            The initial cells, or `None` for random ones.

        random_state:
            The seed or generator of the random initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The class is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NilpotentAutomatonSystem
        >>> len(NilpotentAutomatonSystem(3).prepare().system_.interactions[0].neighborhood.offsets)
        4
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'cellular automaton', 'nilpotent')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 6
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        n: int = 4,
        *,
        size: int = 80,
        density: float = 0.7,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.size = size
        self.density = density
        self.initial = initial
        self.random_state = random_state

    @staticmethod
    def _nilpotent(*values: Array) -> Array:
        """Return 0 on a window of ones or at the left end of a run of ones, and the cell otherwise.

        The values are those of the offsets `-1, 0, ..., n - 1`, so the cell itself is the second.
        """
        window = np.stack(values)
        left, center = window[0], window[1]
        killed = np.all(window == 1, axis=0) | ((left == 0) & (center == 1))
        return np.where(killed, 0.0, center)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.n < MIN_NILPOTENT_CLASS:
            error_msg = f'The nilpotency class of the automaton is at least 2, but it is {self.n}.'
            raise ParameterError(error_msg)
        values = (
            _random_cells(self.size, self.density, self.random_state)
            if self.initial is None
            else np.asarray(self.initial, dtype=float)
        )
        return _line_system(self.label, self._nilpotent, range(-1, self.n), values)


class InvolutiveAutomatonSystem(BaseSystem):
    """A binary automaton on a ring that is its own inverse, with the minimal memory set `{-1, 0, ..., n}`.

    A cell at the left end of a run of exactly `n` ones, the window `alpha = 0 1^n 0`, becomes 0, a cell just before a
    run of exactly `n - 1` ones, the window `beta = 0 0 1^(n - 1) 0`, becomes 1, and every other cell keeps its value
    (Ceccherini-Silberstein and Coornaert 2023, Exercise 1.36). The two moves undo each other, so two steps return every
    configuration to itself, and the automaton is invertible with `τ^-1 = τ`. The system has the cells `cells.alive`.

    Args:
        n:
            The length of the runs of ones, at least 3.

        size:
            The number of cells of the ring.

        density:
            The probability that a cell starts at 1, when no initial cells are given.

        initial:
            The initial cells, or `None` for random ones.

        random_state:
            The seed or generator of the random initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The run length is below 3.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import InvolutiveAutomatonSystem
        >>> len(InvolutiveAutomatonSystem(3).prepare().system_.interactions[0].neighborhood.offsets)
        5
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'cellular automaton', 'invertible', 'involution')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        n: int = 3,
        *,
        size: int = 80,
        density: float = 0.6,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.size = size
        self.density = density
        self.initial = initial
        self.random_state = random_state

    @staticmethod
    def _involution(*values: Array) -> Array:
        """Return the cell with `alpha = 0 1^n 0` sent to 0, `beta = 0 0 1^(n - 1) 0` sent to 1, and other windows kept.

        The values are those of the offsets `-1, 0, ..., n`, so the cell itself is the second.
        """
        window = np.stack(values)
        run = window.shape[0] - 2
        alpha = np.asarray([0, *([1] * run), 0])[:, None]
        beta = np.asarray([0, 0, *([1] * (run - 1)), 0])[:, None]
        center = window[1]
        return np.where(np.all(window == alpha, axis=0), 0.0, np.where(np.all(window == beta, axis=0), 1.0, center))

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.n < MIN_INVOLUTION_RUN:
            error_msg = f'The run length of the involutive automaton is at least 3, but it is {self.n}.'
            raise ParameterError(error_msg)
        values = (
            _random_cells(self.size, self.density, self.random_state)
            if self.initial is None
            else np.asarray(self.initial, dtype=float)
        )
        return _line_system(self.label, self._involution, range(-1, self.n + 1), values)
