"""Automata models: life-like rules, noisy rules, discrete diffusion, and an unbounded alphabet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from collections.abc import Callable, Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.interactions import LocalRuleInteraction
from ....core.neighborhoods import BoxNeighborhood, LatticeNeighborhood, MooreNeighborhood, VonNeumannNeighborhood
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....errors import ParameterError
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life

LIFE_RULE = re.compile(r'^B(?P<born>[0-8]*)/S(?P<survive>[0-8]*)$')
PREDECESSORS = LatticeNeighborhood(((-1,), (0,)))


def _parse_life_rule(rule: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the birth and survival counts of a life-like rule in the notation `B36/S23`."""
    match = LIFE_RULE.match(rule.upper())
    if match is None:
        error_msg = f'A life-like rule reads like `B36/S23`, with digits 0 to 8, but it is {rule!r}.'
        raise ParameterError(error_msg)
    return tuple(int(digit) for digit in match['born']), tuple(int(digit) for digit in match['survive'])


def _noisy_rule(number: int) -> Callable[..., Array]:
    """Return the elementary rule of a Wolfram number whose output a uniform draw below the noise flips."""
    mapping = Wolfram(number).table().mapping
    outputs = np.array([mapping[(left, center, right)] for left, center, right in itertools.product((0, 1), repeat=3)])

    def noisy(left: Array, center: Array, right: Array, *, noise: Array | float, random: Array) -> Array:
        output = outputs[(4 * left + 2 * center + right).astype(int)]
        return np.where(random < noise, 1 - output, output).astype(float)

    return noisy


def _diffuse(level: Array, *around: Array, r: Array | float) -> Array:
    """Return the content after the exchange of a share of the difference with every neighbour."""
    return level + r * (sum(around) - len(around) * level)


def _add(left: Array, center: Array) -> Array:
    """Return the sum of a cell and its left neighbour."""
    return left + center


class NoisyElementaryCellElement(BinaryCellElement):
    """Binary cell that holds the probability that the output of its local rule is flipped.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        alive:
            The initial value, 0 or 1.

        noise:
            The probability that the output of the rule is flipped.

    Examples:
        >>> from skcomplex.models.computer_science.automata import NoisyElementaryCellElement
        >>> NoisyElementaryCellElement('cell', noise=0.1).noise
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: float = 0,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual, alive=alive)
        self.noise = noise


class DiffusionCellElement(Element):
    """Cell of a lattice with a content and the share of the difference it exchanges with each neighbour in a step.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial content.

        r:
            The share of the difference with a neighbour exchanged in a step, below `1 / (2 d)` in `d` dimensions.

    Examples:
        >>> from skcomplex.models.computer_science.automata import DiffusionCellElement
        >>> DiffusionCellElement('cell', r=0.2).r
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.r = r


class PascalCellElement(Element):
    """Cell of an automaton over the nonnegative integers.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The initial value, a nonnegative integer.

    Examples:
        >>> from skcomplex.models.computer_science.automata import PascalCellElement
        >>> PascalCellElement('cell', value=3).value
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'life', 'replicator'),
    references=('Bell 1994', 'Charbonneau 2017', 'Downey 2018'),
    defaults={'until': 100, 'dt': 1},
)
def highlife(
    shape: Sequence[int] = (100, 100),
    *,
    rule: str = 'B36/S23',
    density: float = 0.3,
    initial: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return HighLife, `B36/S23`, or any other life-like automaton of the planar zoo by its rule, on a torus.

    HighLife adds birth on six neighbours to Conway's rule, which gives it a small self-replicating pattern. Other
    members of the zoo are Seeds, `B2/S`, Day and Night, `B3678/S34678`, and Life without death, `B3/S012345678`.

    Args:
        shape:
            The number of rows and columns.

        rule:
            The rule in the notation `B<births>/S<survivals>`.

        density:
            The probability that a cell starts alive, when no initial cells are given.

        initial:
            The initial cells, an array of the shape, or `None` for random cells.

        random_state:
            The seed or generator of the random initial cells.

    Returns:
        system:
            The model, with the cells `cells.alive`.

    Raises:
        ParameterError:
            The rule is not in the notation `B<births>/S<survivals>`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import highlife
        >>> sorted(highlife((10, 10), rule='B2/S').interactions[0].rule.rule)
        [(0, 2)]
    """
    born, survive = _parse_life_rule(rule)
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


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'elementary', 'noise', 'stochastic'),
    references=('Lindgren 2024',),
    defaults={'until': 100, 'dt': 1},
)
def noisy_automata(
    rule: int = 18,
    *,
    noise: float = 0.01,
    size: int = 101,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return an elementary cellular automaton whose outputs are flipped at random with a small probability.

    Args:
        rule:
            The Wolfram number from 0 to 255.

        noise:
            The probability that a cell's output is flipped in a step.

        size:
            The number of cells of the ring.

        density:
            The probability that a cell starts alive.

        random_state:
            The seed or generator of the random initial cells.

    Returns:
        system:
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import noisy_automata
        >>> noisy_automata(184, size=11).elements[0].check().prepare().size_
        11
    """
    values = _random_cells((size,), density, random_state)
    space = LatticeSpace(size)
    return System(
        elements=[
            LatticeElement(
                'cells',
                element=NoisyElementaryCellElement,
                space=space,
                values={'alive': values, 'noise': noise},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'rule',
                target='cells',
                state='alive',
                rule=_noisy_rule(rule),
                neighborhood=BoxNeighborhood(),
                stochastic=True,
            ),
        ],
    )


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'diffusion', 'conservation'),
    references=('Downey 2018',),
    defaults={'until': 100, 'dt': 1},
)
def diffusion_automaton(shape: Sequence[int] = (50, 50), *, r: float = 0.1, width: int = 10) -> System:
    """Return a cellular automaton of diffusion, `c' = c + r Σ (c_neighbour - c)`, started from a square of ones.

    The total content is conserved on the torus, and the square spreads into a Gaussian bump.

    Args:
        shape:
            The number of rows and columns of the torus.

        r:
            The share of the difference with a neighbour exchanged in a step.

        width:
            The side of the central square of cells that start with content 1.

    Returns:
        system:
            The model, with the contents `cells.level`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import diffusion_automaton
        >>> float(diffusion_automaton((20, 20), width=4).elements[0].values['level'].sum())
        16.0
    """
    level = np.zeros(tuple(shape))
    rows, columns = (shape[0] - width) // 2, (shape[1] - width) // 2
    level[rows : rows + width, columns : columns + width] = 1.0
    return System(
        elements=[
            LatticeElement(
                'cells',
                element=DiffusionCellElement,
                space=LatticeSpace(tuple(shape)),
                values={'level': level, 'r': r},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'diffusion',
                target='cells',
                state='level',
                rule=_diffuse,
                neighborhood=VonNeumannNeighborhood(center=True),
            ),
        ],
    )


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'integers', 'additive', 'pascal triangle'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 50, 'dt': 1},
)
def unbounded_alphabet_automaton(size: int = 101) -> System:
    """Return an additive automaton over the nonnegative integers, `x'(n) = x(n - 1) + x(n)`, from a single 1.

    Its alphabet has no bound: from a single 1 the cells after `t` steps hold the binomial coefficients `C(t, k)`, so
    every row of Pascal's triangle appears.

    Args:
        size:
            The number of cells of the ring.

    Returns:
        system:
            The model, with the cells `cells.value`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import unbounded_alphabet_automaton
        >>> int(unbounded_alphabet_automaton(9).elements[0].values['value'].sum())
        1
    """
    value = np.zeros(size)
    value[size // 2] = 1.0
    return System(
        elements=[
            LatticeElement('cells', element=PascalCellElement, space=LatticeSpace(size), values={'value': value}),
        ],
        interactions=[
            LocalRuleInteraction('addition', target='cells', state='value', rule=_add, neighborhood=PREDECESSORS),
        ],
    )
