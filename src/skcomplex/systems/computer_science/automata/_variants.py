"""Automata systems: life-like, noisy, diffusion, and unbounded-alphabet rules.

The diffusion is discrete.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram
from ._life_rule import LifeRuleInteraction
from ._pascal_cell import PascalCellElement

LIFE_RULE = re.compile(r'^B(?P<born>[0-8]*)/S(?P<survive>[0-8]*)$')
PREDECESSORS = LatticeNeighborhood(((-1,), (0,)))
ELEMENTARY = LatticeNeighborhood(((-1,), (0,), (1,)))
VON_NEUMANN = RadiusNeighborhood(1, metric='manhattan', center=True)


class DiffusionCellElement(Element):
    """Cell of a lattice with a nonnegative content.

    A `DiscreteDiffusionInteraction` exchanges a share of the difference of contents with every neighbour. With `space`,
    the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        level:
            The initial content, or one content per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DiffusionCellElement
        >>> DiffusionCellElement('cell', level=2.0).level
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        level: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.level = level


class NoisyRuleInteraction(Interaction):
    """Elementary rule of a line of cells whose outputs are flipped at random.

    Every cell takes the output of the elementary rule of its Wolfram number, and a uniform draw below `noise` flips
    it. Without noise it is the elementary automaton, and at noise 1/2 every cell is a fair coin.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        rule:
            The Wolfram number from 0 to 255.

        noise:
            The probability that the output of a cell is flipped in a step.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NoisyRuleInteraction
        >>> NoisyRuleInteraction('rule', target='cells', noise=0.1).noise
        0.1
    """

    def __init__(self, label: str, *, target: Role = 'cells', rule: int = 18, noise: float = 0.01) -> None:
        super().__init__(label)
        self.target = target
        self.rule = rule
        self.noise = noise

    def _get_function(self) -> Callable[..., Array]:
        """Return the elementary rule whose output a uniform draw below the noise flips."""
        mapping = Wolfram(self.rule).get_table().mapping
        outputs = np.array(
            [mapping[(left, center, right)] for left, center, right in itertools.product((0, 1), repeat=3)],
        )
        noise = self.noise

        def noisy(left: Array, center: Array, right: Array, *, random: Array) -> Array:
            output = outputs[(4 * left + 2 * center + right).astype(int)]
            return np.where(random < noise, 1 - output, output).astype(float)

        return noisy

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noisy local rule of the cells as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            alive = context.target.alive
            self.laws_: list[BaseComponent] = [
                UpdateLaw(alive, self._get_function(), neighborhood=ELEMENTARY, label=self.label),
            ]


class DiscreteDiffusionInteraction(Interaction):
    """Discrete diffusion of the contents of cells, `c' = c + r Σ (c_neighbour - c)`.

    Every cell exchanges the share `r` of the difference of contents with each of its four nearest neighbours, so the
    total content is conserved on a torus.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `level` diffuses.

        r:
            The share of the difference with a neighbour exchanged in a step, below `1 / (2 d)` in `d` dimensions.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DiscreteDiffusionInteraction
        >>> DiscreteDiffusionInteraction('diffusion', target='cells', r=0.2).r
        0.2
    """

    def __init__(self, label: str, *, target: Role = 'cells', r: float = 0.1) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_function(self) -> Callable[..., Array]:
        """Return the content after the exchange of a share of the difference with every neighbour."""
        r = self.r

        def diffuse(level: Array, *around: Array) -> Array:
            return level + r * (sum(around) - len(around) * level)

        return diffuse

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the exchange of contents as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            level = context.target.level
            self.laws_: list[BaseComponent] = [
                UpdateLaw(level, self._get_function(), neighborhood=VON_NEUMANN, label=self.label),
            ]


class AdditiveRuleInteraction(Interaction):
    """Additive rule over the nonnegative integers, `x'(n) = x(n - 1) + x(n)`.

    From a single 1 the cells after `t` steps hold the binomial coefficients `C(t, k)`.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `value` the rule updates.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import AdditiveRuleInteraction
        >>> AdditiveRuleInteraction('addition', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _add(left: Array, center: Array) -> Array:
        """Return the sum of a cell and its left neighbour."""
        return left + center

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the sum with the left neighbour as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            value = context.target.value
            self.laws_: list[BaseComponent] = [UpdateLaw(value, self._add, neighborhood=PREDECESSORS, label=self.label)]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The rule is not in the notation `B<births>/S<survivals>`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import HighlifeSystem
        >>> rule = HighlifeSystem((10, 10), rule='B2/S').prepare().system_.interactions[0]
        >>> rule.born, rule.survive
        ((2,), ())
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'life', 'replicator')
    references = ('Bell 1994', 'Charbonneau 2017', 'Downey 2018')
    until = 100
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        rule: str = 'B36/S23',
        density: float = 0.3,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.rule = rule
        self.density = density
        self.initial = initial
        self.random_state = random_state

    def _parse_life_rule(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the birth and survival counts of the rule in the notation `B36/S23`."""
        match = LIFE_RULE.match(self.rule.upper())
        if match is None:
            error_msg = f'A life-like rule reads like `B36/S23`, with digits 0 to 8, but it is {self.rule!r}.'
            raise ParameterError(error_msg)
        return tuple(int(digit) for digit in match['born']), tuple(int(digit) for digit in match['survive'])

    def _get_cells(self) -> Array:
        """Return the initial cells, given or alive with the probability `density`."""
        if self.initial is not None:
            return np.asarray(self.initial, dtype=float)
        random = np.random.default_rng(self.random_state).random(tuple(self.shape))
        return (random < self.density).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells of a torus and their life-like rule."""
        super()._prepare(context)
        born, survive = self._parse_life_rule()
        cells = BinaryCellElement('cells', alive=self._get_cells().ravel(), space=LatticeSpace(tuple(self.shape)))
        rule = LifeRuleInteraction('rule', target='cells', born=born, survive=survive)
        self.system_ = System(self.label, elements=[cells], interactions=[rule])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NoisyAutomataSystem
        >>> NoisyAutomataSystem(184, size=11).prepare().simulate(until=3).get_values('cells.alive').shape
        (4, 11)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.rule = rule
        self.noise = noise
        self.size = size
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of random cells and their noisy rule."""
        super()._prepare(context)
        values = (np.random.default_rng(self.random_state).random(self.size) < self.density).astype(float)
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=values, space=LatticeSpace(self.size))],
            interactions=[NoisyRuleInteraction('rule', target='cells', rule=self.rule, noise=self.noise)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DiffusionAutomatonSystem
        >>> float(DiffusionAutomatonSystem((20, 20), width=4).prepare().system_.elements[0].level.sum())
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.r = r
        self.width = width

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells, a central square of ones, and their diffusion."""
        super()._prepare(context)
        level = np.zeros(tuple(self.shape))
        rows, columns = (self.shape[0] - self.width) // 2, (self.shape[1] - self.width) // 2
        level[rows : rows + self.width, columns : columns + self.width] = 1.0
        self.system_ = System(
            self.label,
            elements=[DiffusionCellElement('cells', level=level.ravel(), space=LatticeSpace(tuple(self.shape)))],
            interactions=[DiscreteDiffusionInteraction('diffusion', target='cells', r=self.r)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import UnboundedAlphabetAutomatonSystem
        >>> int(UnboundedAlphabetAutomatonSystem(9).prepare().system_.elements[0].value.sum())
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells, a single 1, and their additive rule."""
        super()._prepare(context)
        value = np.zeros(self.size)
        value[self.size // 2] = 1.0
        self.system_ = System(
            self.label,
            elements=[PascalCellElement('cells', value=value, space=LatticeSpace(self.size))],
            interactions=[AdditiveRuleInteraction('addition', target='cells')],
        )
