"""Automata systems: life-like, noisy, diffusion, and unbounded-alphabet rules.

The diffusion is discrete.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from collections.abc import Callable, Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import (
    BallNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
)
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ParameterError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life
from ._pascal_cell import PascalCellElement

LIFE_RULE = re.compile(r'^B(?P<born>[0-8]*)/S(?P<survive>[0-8]*)$')
PREDECESSORS = LatticeNeighborhood(((-1,), (0,)))


class DiffusionCellElement(Element):
    """Cell of a lattice with a content and the share of the difference it exchanges.

    The share of the difference is exchanged with each neighbour in a step.

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
        >>> from skcomplex.systems.computer_science.automata import DiffusionCellElement
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


class NoisyElementaryCellElement(BinaryCellElement):
    """Binary cell that holds the probability that its local rule output is flipped.

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
        >>> from skcomplex.systems.computer_science.automata import NoisyElementaryCellElement
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


class HighlifeSystem(BaseSystem):
    """HighLife, `B36/S23`, or any other life-like automaton of the planar zoo.

    The automaton is given by its rule. It runs on a torus. HighLife adds birth on six neighbours to Conway's rule,
    which gives it a small self-replicating pattern. Other members of the zoo are Seeds, `B2/S`, Day and Night,
    `B3678/S34678`, and Life without death, `B3/S012345678`. The system has the cells `cells.alive`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The rule is not in the notation `B<births>/S<survivals>`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import HighlifeSystem
        >>> sorted(HighlifeSystem((10, 10), rule='B2/S').prepare().system_.interactions[0].rule.rule)
        [(0, 2)]
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'life', 'replicator')
    references = ('Bell 1994', 'Charbonneau 2017', 'Downey 2018')
    until = 100
    dt = 1

    @staticmethod
    def _get_random_cells(
        shape: Sequence[int],
        density: float,
        random_state: int | np.random.Generator | None,
    ) -> Array:
        """Return cells that are 1 with a probability and 0 otherwise."""
        return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        rule: str = 'B36/S23',
        density: float = 0.3,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.rule = rule
        self.density = density
        self.initial = initial
        self.random_state = random_state

    @staticmethod
    def _parse_life_rule(rule: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the birth and survival counts of a life-like rule.

        The rule is in the notation `B36/S23`.
        """
        match = LIFE_RULE.match(rule.upper())
        if match is None:
            error_msg = f'A life-like rule reads like `B36/S23`, with digits 0 to 8, but it is {rule!r}.'
            raise ParameterError(error_msg)
        return tuple(int(digit) for digit in match['born']), tuple(int(digit) for digit in match['survive'])

    def _build_system(self) -> CompositeSystem:
        """Return HighLife, `B36/S23`, or any other life-like automaton of the planar zoo.

        The automaton is given by its rule and runs on a torus.
        """
        born, survive = self._parse_life_rule(self.rule)
        values = (
            HighlifeSystem._get_random_cells(self.shape, self.density, self.random_state)
            if self.initial is None
            else np.asarray(self.initial, dtype=float)
        )
        space = LatticeSpace(tuple(self.shape))
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement('cells', element=BinaryCellElement, space=space, values={'alive': values}, on='sites'),
            ],
            interactions=[
                LocalRuleInteraction(
                    'rule',
                    target='cells',
                    state='alive',
                    rule=life(born, survive),
                    neighborhood=BallNeighborhood(),
                ),
            ],
        )


class NoisyAutomataSystem(BaseSystem):
    """An elementary cellular automaton whose outputs are flipped at random.

    The outputs are flipped with a small probability. The system has the cells `cells.alive`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NoisyAutomataSystem
        >>> NoisyAutomataSystem(184, size=11).prepare().system_.elements[0].prepare().size_
        11
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'elementary', 'noise', 'stochastic')
    references = ('Lindgren 2024',)
    until = 100
    dt = 1

    def __init__(
        self,
        rule: int = 18,
        *,
        noise: float = 0.01,
        size: int = 101,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.rule = rule
        self.noise = noise
        self.size = size
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _get_noisy_rule(number: int) -> Callable[..., Array]:
        """Return the elementary rule of a Wolfram number with noisy outputs.

        A uniform draw below the noise flips its output.
        """
        mapping = Wolfram(number).get_table().mapping
        outputs = np.array(
            [mapping[(left, center, right)] for left, center, right in itertools.product((0, 1), repeat=3)],
        )

        def noisy(left: Array, center: Array, right: Array, *, noise: Array | float, random: Array) -> Array:
            output = outputs[(4 * left + 2 * center + right).astype(int)]
            return np.where(random < noise, 1 - output, output).astype(float)

        return noisy

    def _build_system(self) -> CompositeSystem:
        """Return an elementary cellular automaton whose outputs are flipped at random.

        The outputs are flipped with a small probability.
        """
        values = HighlifeSystem._get_random_cells((self.size,), self.density, self.random_state)
        space = LatticeSpace(self.size)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'cells',
                    element=NoisyElementaryCellElement,
                    space=space,
                    values={'alive': values, 'noise': self.noise},
                    on='sites',
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'rule',
                    target='cells',
                    state='alive',
                    rule=self._get_noisy_rule(self.rule),
                    neighborhood=BallNeighborhood(n_dim=1, center='inline'),
                    stochastic=True,
                ),
            ],
        )


class DiffusionAutomatonSystem(BaseSystem):
    """A cellular automaton of diffusion, `c' = c + r Σ (c_neighbour - c)`.

    It starts from a square of ones. The total content is conserved on the torus, and the square spreads into a Gaussian
    bump. The system has the contents `cells.level`.

    Args:
        shape:
            The number of rows and columns of the torus.

        r:
            The share of the difference with a neighbour exchanged in a step.

        width:
            The side of the central square of cells that start with content 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DiffusionAutomatonSystem
        >>> float(DiffusionAutomatonSystem((20, 20), width=4).prepare().system_.elements[0].values['level'].sum())
        16.0
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'diffusion', 'conservation')
    references = ('Downey 2018',)
    until = 100
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (50, 50),
        *,
        r: float = 0.1,
        width: int = 10,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.r = r
        self.width = width

    @staticmethod
    def _diffuse(level: Array, *around: Array, r: Array | float) -> Array:
        """Return the content after the exchange of a share of the difference.

        The share of the difference is exchanged with every neighbour.
        """
        return level + r * (sum(around) - len(around) * level)

    def _build_system(self) -> CompositeSystem:
        """Return a cellular automaton of diffusion, `c' = c + r Σ (c_neighbour - c)`.

        It starts from a square of ones.
        """
        level = np.zeros(tuple(self.shape))
        rows, columns = (self.shape[0] - self.width) // 2, (self.shape[1] - self.width) // 2
        level[rows : rows + self.width, columns : columns + self.width] = 1.0
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'cells',
                    element=DiffusionCellElement,
                    space=LatticeSpace(tuple(self.shape)),
                    values={'level': level, 'r': self.r},
                    on='sites',
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'diffusion',
                    target='cells',
                    state='level',
                    rule=self._diffuse,
                    neighborhood=BallNeighborhood(metric='manhattan', center='first'),
                ),
            ],
        )


class UnboundedAlphabetAutomatonSystem(BaseSystem):
    """An additive automaton over the nonnegative integers, `x'(n) = x(n - 1) + x(n)`.

    It starts from a single 1.

    Its alphabet has no bound: from a single 1 the cells after `t` steps hold the binomial coefficients `C(t, k)`, so
    every row of Pascal's triangle appears. Reduced modulo 2, the rule is the mirror image of the additive automaton
    `x(n) + x(n + 1)` of Ceccherini-Silberstein and Coornaert (2023, Example 3.3.8), which this system lifts to the
    nonnegative integers. The system has the cells `cells.value`.

    Args:
        size:
            The number of cells of the ring.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import UnboundedAlphabetAutomatonSystem
        >>> int(UnboundedAlphabetAutomatonSystem(9).prepare().system_.elements[0].values['value'].sum())
        1
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'integers', 'additive', 'pascal triangle')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 50
    dt = 1

    def __init__(
        self,
        size: int = 101,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size

    @staticmethod
    def _add(left: Array, center: Array) -> Array:
        """Return the sum of a cell and its left neighbour."""
        return left + center

    def _build_system(self) -> CompositeSystem:
        """Return an additive automaton over the nonnegative integers.

        Its rule is `x'(n) = x(n - 1) + x(n)`. It starts from a single 1.
        """
        value = np.zeros(self.size)
        value[self.size // 2] = 1.0
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'cells',
                    element=PascalCellElement,
                    space=LatticeSpace(self.size),
                    values={'value': value},
                    on='sites',
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'addition',
                    target='cells',
                    state='value',
                    rule=self._add,
                    neighborhood=PREDECESSORS,
                ),
            ],
        )
