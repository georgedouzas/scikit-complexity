"""Subshifts beyond finite type, and the adding machine.

The subshifts are the S-gap, even, context-free, and Morse subshifts.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sage.all__sagemath_symbolics import ZZ

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.rewriting import ProductionLaw, RewritingInteraction, WordElement
from ._binary_cell import BinaryCellElement

OUTSIDE = -1.0
GAP_CAP = 10
BURN_IN = 3
MIN_BASE = 2
SYMBOLS = 3
BALANCED_BLOCK = 0.5


class DigitCellElement(Element):
    """Cell holding a digit of a q-adic integer, from 0 to `base - 1`.

    The digit is the label of a vertex of the q-ary tree. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        digit:
            The initial digit, or one digit per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DigitCellElement
        >>> DigitCellElement('cell', digit=1).digit
        1
    """

    def __init__(
        self,
        label: str,
        *,
        digit: Annotated[float | Array, ZZ] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.digit = digit


class TernaryCellElement(Element):
    """Cell of a subshift over the alphabet `{0, 1, 2}`.

    An example is the context-free subshift. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        symbol:
            The initial symbol, 0, 1, or 2, or one symbol per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TernaryCellElement
        >>> TernaryCellElement('cell', symbol=2).symbol
        2
    """

    def __init__(
        self,
        label: str,
        *,
        symbol: Annotated[float | Array, {0, 1, 2}] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.symbol = symbol


class GapShiftInteraction(Interaction):
    """Shift of a window of a binary configuration whose gaps between ones are independent.

    Every step moves the window one cell to the left. The cell entering on the right is a one with the hazard of the
    gap since the last one, the probability that a gap of that length ends there.

    Args:
        label:
            The label of the interaction.

        target:
            The cells of the window, whose state `alive` the shift moves.

        hazards:
            The probability that a gap ends at every length, from 0.

        size:
            The number of cells of the window.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GapShiftInteraction
        >>> GapShiftInteraction('shift', target='cells', hazards=(0.5, 1.0), size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'cells',
        hazards: Sequence[float] = (1.0,),
        size: int = 120,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.hazards = hazards
        self.size = size

    def _get_function(self) -> Callable[..., Array]:
        """Return the shift that reads the cell to the right, or draws a new cell at the right end."""
        hazards = np.asarray(self.hazards, dtype=float)

        def shift(*values: Array, random: Array) -> Array:
            window = np.stack(values)
            right, inside = window[-1], window[1:-1]
            ones = inside[::-1] == 1
            gap = np.where(ones.any(axis=0), ones.argmax(axis=0), inside.shape[0])
            drawn = (random < hazards[np.minimum(gap, hazards.size - 1)]).astype(float)
            return np.where(right == OUTSIDE, drawn, right)

        return shift

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the shift, which reads the whole window and the cell to the right, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            window = LatticeNeighborhood(tuple((offset,) for offset in range(1 - self.size, 2)))
            alive = context.target.alive
            self.laws_: list[BaseComponent] = [
                UpdateLaw(alive, self._get_function(), neighborhood=window, label=self.label),
            ]


class ContextFreeMoveInteraction(Interaction):
    """Random moves of a ring of symbols that stay in the context-free subshift.

    Every step proposes to change every cell, in a random order, to one of the two other symbols, and rejects the
    changes that would leave a block `0 1^h 2^k 0` with `h != k`. The moves sample the admissible rings uniformly.

    Args:
        label:
            The label of the interaction.

        target:
            The cells of the ring, whose state `symbol` the moves change.

        size:
            The number of cells of the ring.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import ContextFreeMoveInteraction
        >>> ContextFreeMoveInteraction('moves', target='cells', size=12).size
        12
    """

    def __init__(self, label: str, *, target: Role = 'cells', size: int = 60) -> None:
        super().__init__(label)
        self.target = target
        self.size = size

    @staticmethod
    def _is_balanced(word: Array) -> bool:
        """Return whether a word over `{1, 2}` between two zeros is no block `1^h 2^k` with `h != k`."""
        if np.any(np.diff(word) < 0):
            return True
        ones = int(np.sum(word == 1))
        return ones == word.size - ones

    @staticmethod
    def _is_context_free(ring: Array) -> bool:
        """Return whether a ring of symbols holds no block `0 1^h 2^k 0` with `h != k` between consecutive zeros."""
        zeros = np.flatnonzero(ring == 0)
        if zeros.size == 0:
            return True
        rolled = np.roll(ring, -int(zeros[0]))
        bounds = [*(zeros - zeros[0]).tolist(), ring.size]
        return all(
            ContextFreeMoveInteraction._is_balanced(rolled[start + 1 : end])
            for start, end in itertools.pairwise(bounds)
        )

    @staticmethod
    def _get_move(*values: Array, random: Array) -> Array:
        """Return the cell changed to another random symbol, unless the ring would leave the subshift."""
        window = np.stack(values)
        size = (window.shape[0] + 1) // 2
        center = window[size - 1]
        proposals = (center + 1 + np.floor(2 * random)) % SYMBOLS
        following = center.copy()
        for row in range(window.shape[1]):
            ring = window[size - 1 :, row].copy()
            ring[0] = proposals[row]
            if ContextFreeMoveInteraction._is_context_free(ring):
                following[row] = proposals[row]
        return following

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the moves, which read every cell of the ring from both sides, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            ring = LatticeNeighborhood(tuple((offset,) for offset in range(1 - self.size, self.size)))
            symbol = context.target.symbol
            self.laws_: list[BaseComponent] = [
                UpdateLaw(symbol, self._get_move, neighborhood=ring, schedule='random-sequential', label=self.label),
            ]


class CarryInteraction(Interaction):
    """Addition of 1 to the digits of a q-adic integer, the lowest first.

    A digit turns by the cycle `0 → 1 → ... → q - 1 → 0` when every lower digit is `q - 1`, the carry, and keeps its
    value otherwise. The lowest digit reads `q - 1` beyond the digits, so it turns at every step.

    Args:
        label:
            The label of the interaction.

        target:
            The cells of the digits, whose state `digit` the addition changes.

        base:
            The base `q`.

        digits:
            The number of digits.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import CarryInteraction
        >>> CarryInteraction('carry', target='cells', base=3).base
        3
    """

    def __init__(self, label: str, *, target: Role = 'cells', base: int = 2, digits: int = 6) -> None:
        super().__init__(label)
        self.target = target
        self.base = base
        self.digits = digits

    def _get_function(self) -> Callable[..., Array]:
        """Return the digit plus 1 modulo the base when every lower digit is `base - 1`, and the digit otherwise."""
        base = self.base

        def carry(*digits: Array) -> Array:
            window = np.stack(digits)
            carried = np.all(window[:-1] == base - 1, axis=0)
            return np.where(carried, (window[-1] + 1) % base, window[-1])

        return carry

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the carry, which reads every lower digit, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            lower = LatticeNeighborhood(tuple((offset,) for offset in range(1 - self.digits, 1)))
            digit = context.target.digit
            self.laws_: list[BaseComponent] = [
                UpdateLaw(digit, self._get_function(), neighborhood=lower, label=self.label),
            ]


class GapSubshiftSystem(BaseSystem):
    """The shift on the S-gap subshift.

    The S-gap subshift holds the binary sequences whose runs of zeros between ones lie in `S`.

    With `S` the even numbers it is the even subshift, strictly sofic and not of finite type (Ceccherini-Silberstein and
    Coornaert 2023, Exercises 1.85 and 1.92). The cells are a window of a configuration drawn from the measure of
    maximal entropy, whose gaps are independent with the probability `λ^-(h + 1)` of the gap `h`, where `λ` solves
    `sum_(h in S) λ^-(h + 1) = 1` and `log λ` is the entropy. Every step shifts the window one cell to the left, and the
    cell entering on the right is a one with the probability that the gap since the last one ends there. A gap longer
    than the window counts as the window. Gaps are taken below ten times the size. The system has the cells
    `cells.alive`.

    Args:
        size:
            The number of cells of the window.

        gaps:
            The allowed numbers of zeros between consecutive ones, the set `S`, or `None` for the even numbers.

        random_state:
            The seed or generator of the initial window.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            No allowed gap lies below ten times the size.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GapSubshiftSystem
        >>> cells = GapSubshiftSystem(40).prepare().system_.elements[0].alive
        >>> runs = [len(run) for run in ''.join(str(int(cell)) for cell in cells).strip('0').split('1') if run]
        >>> all(run % 2 == 0 for run in runs)
        True
    """

    field = 'computer_science/automata'
    topics = ('subshift', 'S-gap', 'even shift', 'renewal', 'sofic')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 100
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        size: int = 120,
        *,
        gaps: Sequence[int] | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.gaps = gaps
        self.random_state = random_state

    @staticmethod
    def _get_gap_lengths(size: int, gaps: Sequence[int] | None) -> npt.NDArray[np.bool_]:
        """Return whether every gap length below the cap is allowed.

        By default the even lengths are allowed.
        """
        lengths = np.arange(GAP_CAP * size)
        if gaps is None:
            return lengths % 2 == 0
        allowed = np.zeros(lengths.size, dtype=bool)
        chosen = np.asarray([gap for gap in gaps if 0 <= gap < lengths.size], dtype=np.intp)
        allowed[chosen] = True
        if not allowed.any():
            error_msg = f'The S-gap subshift needs at least one allowed gap below {lengths.size}, but got {gaps!r}.'
            raise ParameterError(error_msg)
        return allowed

    @staticmethod
    def _get_gap_law(allowed: npt.NDArray[np.bool_]) -> tuple[float, Array, Array]:
        """Return the growth and the gap law of the maximal entropy measure.

        The law is given by the growth `λ`, the gap probabilities `λ^-(h + 1)`, and the hazards.
        """
        lengths = np.flatnonzero(allowed)

        def excess(growth: float) -> float:
            return float(np.sum(growth ** -(lengths + 1.0))) - 1.0

        growth = 1.0 if excess(1.0) <= 0 else float(scipy.optimize.brentq(excess, 1.0, 2.0))
        probabilities = np.where(allowed, growth ** -(np.arange(allowed.size) + 1.0), 0.0)
        probabilities /= probabilities.sum()
        survival = probabilities[::-1].cumsum()[::-1]
        hazards = np.divide(probabilities, survival, out=np.ones_like(probabilities), where=survival > 0)
        return growth, probabilities, hazards

    @staticmethod
    def _get_renewal_cells(size: int, probabilities: Array, random_state: int | np.random.Generator | None) -> Array:
        """Return a window of a configuration whose gaps between ones are independent.

        The gaps are drawn independently, after a burn-in.
        """
        generator = np.random.default_rng(random_state)
        cells: list[float] = []
        while len(cells) < BURN_IN * size:
            cells += [1.0] + [0.0] * int(generator.choice(probabilities.size, p=probabilities))
        return np.asarray(cells[-size:])

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a window drawn from the maximal entropy measure and its shift."""
        super()._prepare(context)
        _, probabilities, hazards = self._get_gap_law(self._get_gap_lengths(self.size, self.gaps))
        cells = self._get_renewal_cells(self.size, probabilities, self.random_state)
        space = LatticeSpace(self.size, boundary='fixed', fill=OUTSIDE)
        shift = GapShiftInteraction('shift', target='cells', hazards=tuple(hazards.tolist()), size=self.size)
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=cells, space=space)],
            interactions=[shift],
        )


class ContextFreeSubshiftSystem(BaseSystem):
    """A Markov chain on the context-free subshift.

    The subshift holds the ternary rings where every block `0 1^h 2^k 0` has `h = k`.

    The subshift is not sofic, since a finite graph cannot match the counts of ones and twos (Ceccherini-Silberstein and
    Coornaert 2023, Exercise 6.80). Every step proposes to change every cell of the ring, in a random order, to one of
    the two other symbols, and rejects the changes that would leave the subshift, so the chain samples its admissible
    rings uniformly. It starts from blocks `0 1^h 2^h`, with `h` drawn from a geometric law. The system has the symbols
    `cells.symbol`.

    Args:
        size:
            The number of cells of the ring.

        initial:
            The initial symbols, an admissible ring, or `None` for random balanced blocks.

        random_state:
            The seed or generator of the initial blocks.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The initial ring is not admissible.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import ContextFreeSubshiftSystem
        >>> ContextFreeSubshiftSystem(12).prepare().system_.elements[0].symbol.tolist()
        [0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]
    """

    field = 'computer_science/automata'
    topics = ('subshift', 'context-free', 'Monte Carlo', 'not sofic')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 50
    dt = 1

    def __init__(
        self,
        size: int = 60,
        *,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.initial = initial
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of balanced blocks and its random moves."""
        super()._prepare(context)
        if self.initial is None:
            generator = np.random.default_rng(self.random_state)
            blocks: list[float] = []
            while len(blocks) < self.size:
                height = int(generator.geometric(BALANCED_BLOCK))
                blocks += [0.0] + [1.0] * height + [2.0] * height
            symbols = np.zeros(self.size)
            cut = max(index for index in range(self.size + 1) if index == len(blocks) or blocks[index] == 0)
            symbols[:cut] = blocks[:cut]
        else:
            symbols = np.asarray(self.initial, dtype=float)
        if not ContextFreeMoveInteraction._is_context_free(symbols):
            error_msg = f'The initial ring {symbols.tolist()} holds a block 0 1^h 2^k 0 with h != k.'
            raise ParameterError(error_msg)
        self.system_ = System(
            self.label,
            elements=[TernaryCellElement('cells', symbol=symbols, space=LatticeSpace(self.size))],
            interactions=[ContextFreeMoveInteraction('moves', target='cells', size=self.size)],
        )


class MorseSubshiftSystem(BaseSystem):
    """The Thue-Morse substitution `0 → 01`, `1 → 10`.

    Its words make the language of the Morse subshift. Generation `n` from `0` is the prefix `T(0) ... T(2^n - 1)` of
    the Thue-Morse sequence, and the Morse subshift is the orbit closure under the shift of the configuration
    `x(n) = T(n)` for `n >= 0` and `T(-n - 1)` for `n < 0`, an infinite minimal subshift that is not sofic, with zero
    entropy (Ceccherini-Silberstein and Coornaert 2023, Exercises 3.51 and 3.52). Its words of every length are the
    factors of the generations, which number at most a multiple of their length. The system has the word `morse` and its
    length `morse.size`.

    Args:
        axiom:
            The word at generation 0, `0` for the Thue-Morse sequence and `1` for its mirror.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import MorseSubshiftSystem
        >>> MorseSubshiftSystem().prepare().simulate(until=3).get_values('morse.size').tolist()
        [1.0, 2.0, 4.0, 8.0]
    """

    field = 'computer_science/automata'
    topics = ('subshift', 'Thue-Morse', 'substitution', 'minimal', 'rewriting system')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1

    def __init__(
        self,
        axiom: str = '0',
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.axiom = axiom

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the word and its substitution."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[WordElement('morse', axiom=self.axiom, alphabet='01')],
            interactions=[
                RewritingInteraction(
                    'doubling',
                    target='morse',
                    laws=[ProductionLaw('0', '01'), ProductionLaw('1', '10')],
                ),
            ],
        )


class AddingMachineSystem(BaseSystem):
    """The adding machine `r → r + 1` on the q-adic integers.

    It is the odometer acting on the q-ary tree.

    The cells hold the digits of `r`, the lowest first, the first `digits` letters of its infinite word, a vertex of the
    q-ary tree. Every step adds 1: the lowest digit turns by the cycle `0 → 1 → ... → q - 1 → 0`, and a digit turns when
    every lower digit is `q - 1`, the carry (Ceccherini-Silberstein and Coornaert 2023, Exercise 6.59). The machine is
    an automorphism of the tree of infinite order whose orbits are dense, and on the first `digits` levels it visits all
    `q^digits` vertices in turn. The lowest digit reads `q - 1` beyond the segment, the carry of `+1`. The system has
    the digits `cells.digit`.

    Args:
        digits:
            The number of digits kept, the depth of the tree.

        base:
            The base `q`, at least 2.

        initial:
            The initial digits, the lowest first, or `None` for 0.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The base is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import AddingMachineSystem
        >>> AddingMachineSystem(3).prepare().simulate(until=5).get_values('cells.digit')[-1].tolist()
        [1.0, 0.0, 1.0]
    """

    field = 'computer_science/automata'
    topics = ('odometer', 'q-adic integers', 'self-similar group', 'tree automorphism')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 64
    dt = 1

    def __init__(
        self,
        digits: int = 6,
        *,
        base: int = 2,
        initial: npt.ArrayLike | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.digits = digits
        self.base = base
        self.initial = initial

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the digits and the addition of 1."""
        super()._prepare(context)
        if self.base < MIN_BASE:
            error_msg = f'The base of the adding machine is at least 2, but it is {self.base}.'
            raise ParameterError(error_msg)
        values = np.zeros(self.digits) if self.initial is None else np.asarray(self.initial, dtype=float)
        space = LatticeSpace(self.digits, boundary='fixed', fill=self.base - 1)
        self.system_ = System(
            self.label,
            elements=[DigitCellElement('cells', digit=values, space=space)],
            interactions=[CarryInteraction('carry', target='cells', base=self.base, digits=self.digits)],
        )
