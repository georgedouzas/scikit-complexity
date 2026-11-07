"""Subshifts beyond finite type, the S-gap, even, context-free, and Morse subshifts, and the adding machine."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement, WordElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.laws import ProductionLaw
from ....core.components.spaces import LatticeNeighborhood, LatticeSpace
from ....core.components.system import System
from ....core.engine.compile import RewritingCompiler
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ._binary_cell import BinaryCellElement


class DigitCellElement(Element):
    """Cell holding a digit of a q-adic integer, from 0 to `base - 1`, the label of a vertex of the q-ary tree.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        digit:
            The initial digit.

        base:
            The base `q` of the digits, at least 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DigitCellElement
        >>> DigitCellElement('cell', digit=1).base
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        digit: Annotated[float, StateSpec(domain=ZZ)] = 0,
        base: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.digit = digit
        self.base = base


class MorseElement(WordElement):
    """Word rewritten by the Thue-Morse substitution `0 → 01`, `1 → 10`.

    From the axiom `0`, generation `n` is the word `a_n = T(0) T(1) ... T(2^n - 1)` of the Thue-Morse sequence, whose
    term `T(k)` is the parity of the number of ones in the binary expansion of `k`.

    Args:
        label:
            The label of the word.

        axiom:
            The word at generation 0.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import MorseElement
        >>> [law.successor for law in MorseElement('morse', axiom='0')._get_laws(None)]
        ['01', '10']
    """

    def _get_laws(self, s: SimpleNamespace) -> list[ProductionLaw]:
        """Return the productions of the Thue-Morse substitution.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The productions of the word.
        """
        del s
        return [ProductionLaw('0', '01'), ProductionLaw('1', '10')]


class TernaryCellElement(Element):
    """Cell of a subshift over the alphabet `{0, 1, 2}`, such as the context-free subshift.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol, 0, 1, or 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import TernaryCellElement
        >>> TernaryCellElement('cell', symbol=2).symbol
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1, 2]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol


OUTSIDE = -1.0
GAP_CAP = 10
BURN_IN = 3
MIN_BASE = 2
SYMBOLS = 3
BALANCED_BLOCK = 0.5


def _gap_lengths(size: int, gaps: Sequence[int] | None) -> npt.NDArray[np.bool_]:
    """Return whether every gap length below the cap is allowed, the even lengths by default."""
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


def _gap_law(allowed: npt.NDArray[np.bool_]) -> tuple[float, Array, Array]:
    """Return the growth `λ`, the gap probabilities `λ^-(h + 1)`, and the hazards of the measure of maximal entropy."""
    lengths = np.flatnonzero(allowed)

    def excess(growth: float) -> float:
        return float(np.sum(growth ** -(lengths + 1.0))) - 1.0

    growth = 1.0 if excess(1.0) <= 0 else float(scipy.optimize.brentq(excess, 1.0, 2.0))
    probabilities = np.where(allowed, growth ** -(np.arange(allowed.size) + 1.0), 0.0)
    probabilities /= probabilities.sum()
    survival = probabilities[::-1].cumsum()[::-1]
    hazards = np.divide(probabilities, survival, out=np.ones_like(probabilities), where=survival > 0)
    return growth, probabilities, hazards


def _renewal_cells(size: int, probabilities: Array, random_state: int | np.random.Generator | None) -> Array:
    """Return a window of a configuration whose gaps between ones are drawn independently, after a burn-in."""
    generator = np.random.default_rng(random_state)
    cells: list[float] = []
    while len(cells) < BURN_IN * size:
        cells += [1.0] + [0.0] * int(generator.choice(probabilities.size, p=probabilities))
    return np.asarray(cells[-size:])


def _gap_shift(*values: Array, random: Array, hazards: Array) -> Array:
    """Return the cell to the right, or at the right end a new cell drawn from the gap since the last one."""
    window = np.stack(values)
    right, inside = window[-1], window[1:-1]
    ones = inside[::-1] == 1
    gap = np.where(ones.any(axis=0), ones.argmax(axis=0), inside.shape[0])
    drawn = (random < hazards[np.minimum(gap, hazards.size - 1)]).astype(float)
    return np.where(right == OUTSIDE, drawn, right)


def _is_balanced(word: Array) -> bool:
    """Return whether a word over `{1, 2}` between two zeros is no block `1^h 2^k` with `h != k`."""
    if np.any(np.diff(word) < 0):
        return True
    ones = int(np.sum(word == 1))
    return ones == word.size - ones


def _is_context_free(ring: Array) -> bool:
    """Return whether a ring of symbols holds no block `0 1^h 2^k 0` with `h != k` between consecutive zeros."""
    zeros = np.flatnonzero(ring == 0)
    if zeros.size == 0:
        return True
    rolled = np.roll(ring, -int(zeros[0]))
    bounds = [*(zeros - zeros[0]).tolist(), ring.size]
    return all(_is_balanced(rolled[start + 1 : end]) for start, end in itertools.pairwise(bounds))


def _context_free_move(*values: Array, random: Array) -> Array:
    """Return the cell changed to another symbol at random, unless the ring would leave the context-free subshift."""
    window = np.stack(values)
    size = (window.shape[0] + 1) // 2
    center = window[size - 1]
    proposals = (center + 1 + np.floor(2 * random)) % SYMBOLS
    following = center.copy()
    for row in range(window.shape[1]):
        ring = window[size - 1 :, row].copy()
        ring[0] = proposals[row]
        if _is_context_free(ring):
            following[row] = proposals[row]
    return following


def _carry(*digits: Array, base: Array | float) -> Array:
    """Return the digit plus 1 modulo the base when every lower digit is `base - 1`, and the digit otherwise."""
    window = np.stack(digits)
    carried = np.all(window[:-1] == base - 1, axis=0)
    return np.where(carried, (window[-1] + 1) % base, window[-1])


def _density_observables(system: System) -> list[ObservableSpec]:
    """Return the density of ones of the cells."""
    return [ObservableSpec('density', group='cells', variable='alive')]


def _whole_ring(size: int) -> LatticeNeighborhood:
    """Return the offsets from `-(size - 1)` to `size - 1`, which read every cell of a ring from both sides."""
    return LatticeNeighborhood(tuple((offset,) for offset in range(1 - size, size)))


@register_system(
    field='computer_science/automata',
    tags=('subshift', 'S-gap', 'even shift', 'renewal', 'sofic'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 100, 'dt': 1},
    observables=_density_observables,
)
def gap_subshift(
    size: int = 120,
    *,
    gaps: Sequence[int] | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the shift on the S-gap subshift, the binary sequences whose runs of zeros between ones lie in `S`.

    With `S` the even numbers it is the even subshift, strictly sofic and not of finite type
    (Ceccherini-Silberstein and Coornaert 2023, Exercises 1.85 and 1.92). The cells are a window of a configuration
    drawn from the measure of maximal entropy, whose gaps are independent with the probability `λ^-(h + 1)` of the gap
    `h`, where `λ` solves `sum_(h in S) λ^-(h + 1) = 1` and `log λ` is the entropy. Every step shifts the window one
    cell to the left, and the cell entering on the right is a one with the probability that the gap since the last one
    ends there. A gap longer than the window counts as the window. Gaps are taken below ten times the size.

    Args:
        size:
            The number of cells of the window.

        gaps:
            The allowed numbers of zeros between consecutive ones, the set `S`, or `None` for the even numbers.

        random_state:
            The seed or generator of the initial window.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Raises:
        ParameterError:
            No allowed gap lies below ten times the size.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import gap_subshift
        >>> cells = gap_subshift(40).elements[0].values['alive']
        >>> runs = [len(run) for run in ''.join(str(int(cell)) for cell in cells).strip('0').split('1') if run]
        >>> all(run % 2 == 0 for run in runs)
        True
    """
    _, probabilities, hazards = _gap_law(_gap_lengths(size, gaps))
    cells = _renewal_cells(size, probabilities, random_state)
    neighborhood = LatticeNeighborhood(tuple((offset,) for offset in range(1 - size, 2)))
    return System(
        elements=[
            LatticeElement(
                'cells',
                element=BinaryCellElement,
                space=LatticeSpace(size, boundary='fixed', fill=OUTSIDE),
                values={'alive': cells},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'shift',
                target='cells',
                state='alive',
                rule=functools.partial(_gap_shift, hazards=hazards),
                neighborhood=neighborhood,
                stochastic=True,
            ),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('subshift', 'context-free', 'Monte Carlo', 'not sofic'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 50, 'dt': 1},
)
def context_free_subshift(
    size: int = 60,
    *,
    initial: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a Markov chain on the context-free subshift, ternary rings where every block `0 1^h 2^k 0` has `h = k`.

    The subshift is not sofic, since a finite graph cannot match the counts of ones and twos
    (Ceccherini-Silberstein and Coornaert 2023, Exercise 6.80). Every step proposes to change every cell of the ring, in
    a random order, to one of the two other symbols, and rejects the changes that would leave the subshift, so the
    chain samples its admissible rings uniformly. It starts from blocks `0 1^h 2^h`, with `h` drawn from a geometric
    law.

    Args:
        size:
            The number of cells of the ring.

        initial:
            The initial symbols, an admissible ring, or `None` for random balanced blocks.

        random_state:
            The seed or generator of the initial blocks.

    Returns:
        system:
            The system, with the symbols `cells.symbol`.

    Raises:
        ParameterError:
            The initial ring is not admissible.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import context_free_subshift
        >>> context_free_subshift(12).elements[0].values['symbol'].tolist()
        [0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]
    """
    if initial is None:
        generator = np.random.default_rng(random_state)
        blocks: list[float] = []
        while len(blocks) < size:
            height = int(generator.geometric(BALANCED_BLOCK))
            blocks += [0.0] + [1.0] * height + [2.0] * height
        symbols = np.zeros(size)
        cut = max(index for index in range(size + 1) if index == len(blocks) or blocks[index] == 0)
        symbols[:cut] = blocks[:cut]
    else:
        symbols = np.asarray(initial, dtype=float)
    if not _is_context_free(symbols):
        error_msg = f'The initial ring {symbols.tolist()} holds a block 0 1^h 2^k 0 with h != k.'
        raise ParameterError(error_msg)
    return System(
        elements=[
            LatticeElement('cells', element=TernaryCellElement, space=LatticeSpace(size), values={'symbol': symbols}),
        ],
        interactions=[
            LocalRuleInteraction(
                'moves',
                target='cells',
                state='symbol',
                rule=_context_free_move,
                neighborhood=_whole_ring(size),
                schedule='random-sequential',
                stochastic=True,
            ),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('subshift', 'Thue-Morse', 'substitution', 'minimal', 'rewriting system'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'compiler': RewritingCompiler(), 'until': 10, 'dt': 1},
)
def morse_subshift(axiom: str = '0') -> System:
    """Return the Thue-Morse substitution `0 → 01`, `1 → 10`, whose words make the language of the Morse subshift.

    Generation `n` from `0` is the prefix `T(0) ... T(2^n - 1)` of the Thue-Morse sequence, and the Morse subshift is
    the orbit closure under the shift of the configuration `x(n) = T(n)` for `n >= 0` and `T(-n - 1)` for `n < 0`, an
    infinite minimal subshift that is not sofic, with zero entropy (Ceccherini-Silberstein and Coornaert 2023,
    Exercises 3.51 and 3.52). Its words of every length are the factors of the generations, which number at most a
    multiple of their length.

    Args:
        axiom:
            The word at generation 0, `0` for the Thue-Morse sequence and `1` for its mirror.

    Returns:
        system:
            The system, with the word `morse` and its length `morse.size`.

    Examples:
        >>> from skcomplex.systems import create_simulation
        >>> create_simulation('morse_subshift').prepare().run(until=3).trajectory_.values('morse.size').tolist()
        [1.0, 2.0, 4.0, 8.0]
    """
    return System(elements=[MorseElement('morse', axiom=axiom, alphabet='01')])


@register_system(
    field='computer_science/automata',
    tags=('odometer', 'q-adic integers', 'self-similar group', 'tree automorphism'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 64, 'dt': 1},
)
def adding_machine(
    digits: int = 6,
    *,
    base: int = 2,
    initial: npt.ArrayLike | None = None,
) -> System:
    """Return the adding machine `r → r + 1` on the q-adic integers, the odometer acting on the q-ary tree.

    The cells hold the digits of `r`, the lowest first, the first `digits` letters of its infinite word, a vertex of the
    q-ary tree. Every step adds 1: the lowest digit turns by the cycle `0 → 1 → ... → q - 1 → 0`, and a digit turns
    when every lower digit is `q - 1`, the carry (Ceccherini-Silberstein and Coornaert 2023, Exercise 6.59). The machine
    is an automorphism of the tree of infinite order whose orbits are dense, and on the first `digits` levels it
    visits all `q^digits` vertices in turn. The lowest digit reads `q - 1` beyond the segment, the carry of `+1`.

    Args:
        digits:
            The number of digits kept, the depth of the tree.

        base:
            The base `q`, at least 2.

        initial:
            The initial digits, the lowest first, or `None` for 0.

    Returns:
        system:
            The system, with the digits `cells.digit`.

    Raises:
        ParameterError:
            The base is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import adding_machine
        >>> adding_machine(4).elements[0].prepare().size_
        4
    """
    if base < MIN_BASE:
        error_msg = f'The base of the adding machine is at least 2, but it is {base}.'
        raise ParameterError(error_msg)
    values = np.zeros(digits) if initial is None else np.asarray(initial, dtype=float)
    return System(
        elements=[
            LatticeElement(
                'cells',
                element=DigitCellElement,
                space=LatticeSpace(digits, boundary='fixed', fill=base - 1),
                values={'digit': values, 'base': base},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'carry',
                target='cells',
                state='digit',
                rule=_carry,
                neighborhood=LatticeNeighborhood(tuple((offset,) for offset in range(1 - digits, 1))),
            ),
        ],
    )
