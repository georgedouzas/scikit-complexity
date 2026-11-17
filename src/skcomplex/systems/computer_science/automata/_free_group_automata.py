"""Automata on free groups beyond the Garden of Eden theorem.

They are Muller's automaton, a linear one, and Life in a tree.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ..common.automata import CayleySpace
from ._binary_cell import BinaryCellElement
from ._generator_neighborhood import GeneratorNeighborhood
from ._groups import free_group
from ._life_rule import LifeRuleInteraction

FREE_GENERATORS = GeneratorNeighborhood(('a', 'b'), center=False)
TREE_RANK = 4
MIN_MODULUS = 2


class PairCellElement(Element):
    """Cell of an automaton over the product `H x H` of the integers modulo `n`.

    For a prime `n` the product is `K^2`. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        first:
            The initial first coordinate, an integer from 0 to `n - 1`, or one per cell.

        second:
            The initial second coordinate, an integer from 0 to `n - 1`, or one per cell.

        space:
            The space with one cell on every site, such as a group, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import PairCellElement
        >>> PairCellElement('cell', first=1).first
        1
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[float | Array, ZZ] = 0,
        second: Annotated[float | Array, ZZ] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.first = first
        self.second = second


class MullerRuleInteraction(Interaction):
    """Muller's rule on pairs, `p1(x(ga)) + p2(x(gb)) + p1(x(ga^-1)) + p2(x(gb^-1))`.

    The projections are `p1(h, k) = (h, 0)` and `p2(h, k) = (k, 0)`, and the sum is taken modulo the order of the
    cyclic group, so every image lies in `H x {0}`.

    Args:
        label:
            The label of the interaction.

        target:
            The cells on the free group, whose states `first` and `second` the rule updates.

        modulus:
            The order `n` of the cyclic group `H`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import MullerRuleInteraction
        >>> MullerRuleInteraction('rule', target='cells', modulus=3).modulus
        3
    """

    def __init__(self, label: str, *, target: Role = 'cells', modulus: int = 2) -> None:
        super().__init__(label)
        self.target = target
        self.modulus = modulus

    def _get_function(self) -> Callable[..., tuple[Array, Array]]:
        """Return the rule on the pairs at `a`, `a^-1`, `b`, and `b^-1`."""
        modulus = self.modulus

        def muller(*values: Array) -> tuple[Array, Array]:
            first, second = values[:4], values[4:]
            total = first[0] + first[1] + second[2] + second[3]
            return total % modulus, np.zeros_like(total)

        return muller

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the pairs as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            states = (context.target.first, context.target.second)
            self.laws_: list[BaseComponent] = [
                UpdateLaw(states, self._get_function(), neighborhood=FREE_GENERATORS, label=self.label),
            ]


class SurjectiveLinearRuleInteraction(Interaction):
    """Linear rule on pairs, `q1(x(ga)) + q1(x(ga^-1)) + q2(x(gb)) + q2(x(gb^-1))`.

    The maps are `q1(h, k) = (h, 0)` and `q2(h, k) = (0, h)`, over the field of a prime number of elements.

    Args:
        label:
            The label of the interaction.

        target:
            The cells on the free group, whose states `first` and `second` the rule updates.

        modulus:
            The prime size of the field `K`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SurjectiveLinearRuleInteraction
        >>> SurjectiveLinearRuleInteraction('rule', target='cells', modulus=3).modulus
        3
    """

    def __init__(self, label: str, *, target: Role = 'cells', modulus: int = 2) -> None:
        super().__init__(label)
        self.target = target
        self.modulus = modulus

    def _get_function(self) -> Callable[..., tuple[Array, Array]]:
        """Return the rule on the pairs at `a`, `a^-1`, `b`, and `b^-1`."""
        modulus = self.modulus

        def linear(*values: Array) -> tuple[Array, Array]:
            first = values[:4]
            return (first[0] + first[1]) % modulus, (first[2] + first[3]) % modulus

        return linear

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the pairs as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            states = (context.target.first, context.target.second)
            self.laws_: list[BaseComponent] = [
                UpdateLaw(states, self._get_function(), neighborhood=FREE_GENERATORS, label=self.label),
            ]


class MullerAutomatonSystem(BaseSystem):
    """Muller's automaton on the free group F2, pre-injective but not surjective.

    The alphabet is `A = H x H` for the cyclic group `H` of order `modulus`, and every cell becomes `p1(x(ga)) +
    p2(x(gb)) + p1(x(ga^-1)) + p2(x(gb^-1))`, where `p1(h, k) = (h, 0)` and `p2(h, k) = (k, 0)` (Ceccherini-Silberstein
    and Coornaert 2023, Section 5.11). The image lies in `H x {0}`, so the automaton is not surjective, yet two
    configurations that differ on a finite set have different images, so it is pre-injective, which the Garden of Eden
    theorem forbids on amenable groups. For a prime modulus it is the linear automaton over `K^2` of Section 8.10. Cells
    on the boundary of the ball read 0 outside it. The system has the coordinates `cells.first` and `cells.second`.

    Args:
        radius:
            The radius of the ball of F2.

        modulus:
            The order of `H`, at least 2.

        initial:
            The initial pairs, an array of shape `(sites, 2)`, or `None` for random ones.

        random_state:
            The seed or generator of the random initial pairs.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The modulus is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import MullerAutomatonSystem
        >>> MullerAutomatonSystem(radius=2).prepare().simulate(until=1).get_values('cells.first').shape
        (2, 17)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'free group', 'Garden of Eden')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1
    observables = (
        Observable('first_mean', group='cells', variable='first'),
        Observable('second_mean', group='cells', variable='second'),
    )

    def __init__(
        self,
        *,
        radius: int = 5,
        modulus: int = 2,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.radius = radius
        self.modulus = modulus
        self.initial = initial
        self.random_state = random_state

    def _get_pairs(self, space: BaseSpace) -> Array:
        """Return the initial pairs, one row of two coordinates per site, random modulo the modulus by default."""
        if self.modulus < MIN_MODULUS:
            error_msg = f'The order of the cyclic group H is at least 2, but it is {self.modulus}.'
            raise ParameterError(error_msg)
        size = space.prepare().size_
        if self.initial is not None:
            return np.asarray(self.initial, dtype=float).reshape(size, 2)
        return np.random.default_rng(self.random_state).integers(0, self.modulus, (size, 2)).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of pairs on a ball of the free group of rank 2 and Muller's rule."""
        super()._prepare(context)
        space = CayleySpace(free_group(2), ball=self.radius)
        pairs = self._get_pairs(space)
        self.system_ = System(
            self.label,
            elements=[PairCellElement('cells', first=pairs[:, 0], second=pairs[:, 1], space=space)],
            interactions=[MullerRuleInteraction('rule', target='cells', modulus=self.modulus)],
        )


class SurjectiveLinearAutomatonSystem(BaseSystem):
    """A linear automaton over `K^2` on the free group F2.

    It is surjective but not pre-injective. Every cell becomes `q1(x(ga)) + q1(x(ga^-1)) + q2(x(gb)) + q2(x(gb^-1))`,
    where `q1(h, k) = (h, 0)` and `q2(h, k) = (0, h)`, over the field of a prime number `modulus` of elements
    (Ceccherini-Silberstein and Coornaert 2023, Section 8.11). The configuration `(0, 1)` at the identity and 0
    elsewhere maps to 0, so the automaton is not pre-injective, while a preimage of every configuration is built by
    induction on the word length, so it is surjective, which the Garden of Eden theorem forbids on amenable groups.
    Cells on the boundary of the ball read 0 outside it. The system has the coordinates `cells.first` and
    `cells.second`.

    Args:
        radius:
            The radius of the ball of F2.

        modulus:
            The prime size of the field `K`.

        initial:
            The initial pairs, an array of shape `(sites, 2)`, or `None` for random ones.

        random_state:
            The seed or generator of the random initial pairs.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The modulus is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SurjectiveLinearAutomatonSystem
        >>> SurjectiveLinearAutomatonSystem(radius=2).prepare().simulate(until=1).get_values('cells.first').shape
        (2, 17)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'free group', 'linear', 'Garden of Eden')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1
    observables = (
        Observable('first_mean', group='cells', variable='first'),
        Observable('second_mean', group='cells', variable='second'),
    )

    def __init__(
        self,
        *,
        radius: int = 5,
        modulus: int = 2,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.radius = radius
        self.modulus = modulus
        self.initial = initial
        self.random_state = random_state

    def _get_pairs(self, space: BaseSpace) -> Array:
        """Return the initial pairs, one row of two coordinates per site, random modulo the modulus by default."""
        if self.modulus < MIN_MODULUS:
            error_msg = f'The order of the cyclic group H is at least 2, but it is {self.modulus}.'
            raise ParameterError(error_msg)
        size = space.prepare().size_
        if self.initial is not None:
            return np.asarray(self.initial, dtype=float).reshape(size, 2)
        return np.random.default_rng(self.random_state).integers(0, self.modulus, (size, 2)).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of pairs on a ball of the free group of rank 2 and the linear rule."""
        super()._prepare(context)
        space = CayleySpace(free_group(2), ball=self.radius)
        pairs = self._get_pairs(space)
        self.system_ = System(
            self.label,
            elements=[PairCellElement('cells', first=pairs[:, 0], second=pairs[:, 1], space=space)],
            interactions=[SurjectiveLinearRuleInteraction('rule', target='cells', modulus=self.modulus)],
        )


class LifeInATreeSystem(BaseSystem):
    """The Game of Life on the free group F4, surjective but not pre-injective.

    The free group F4 is a tree of degree 8. The memory set is the identity and the generators `a`, `b`, `c`, `d` with
    their inverses, and a cell is alive after a step when the sum over the memory set is 3, or 4 with the cell alive,
    Conway's rule on the Cayley graph of F4 (Ceccherini-Silberstein and Coornaert 2023, Exercise 5.14). On the tree the
    automaton is surjective but not pre-injective, unlike Life on the plane, which is neither. Cells on the boundary of
    the ball read 0 outside it. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball of F4.

        density:
            The probability that a cell starts alive, when no initial cells are given.

        initial:
            The initial cells, one per site, or `None` for random ones.

        random_state:
            The seed or generator of the random initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LifeInATreeSystem
        >>> LifeInATreeSystem(radius=2).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 65)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'life', 'free group', 'Garden of Eden')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 20
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 4,
        density: float = 0.3,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.radius = radius
        self.density = density
        self.initial = initial
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells on a ball of the free group of rank 4 and the rules of life."""
        super()._prepare(context)
        space = CayleySpace(free_group(TREE_RANK), ball=self.radius)
        size = space.prepare().size_
        if self.initial is None:
            cells = np.asarray(np.random.default_rng(self.random_state).random(size) < self.density, dtype=float)
        else:
            cells = np.asarray(self.initial, dtype=float)
        life = LifeRuleInteraction(
            'life',
            target='cells',
            born=(3,),
            survive=(2, 3),
            neighborhood=GeneratorNeighborhood(space.generators_, center=False),
        )
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=cells, space=space)],
            interactions=[life],
        )
