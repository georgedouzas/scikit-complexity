"""Automata on free groups beyond the Garden of Eden theorem.

They are Muller's automaton, a linear one, and Life in a tree.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import CayleySpace, GeneratorNeighborhood
from ....core.components.systems import BaseSystem, System
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ._binary_cell import BinaryCellElement
from ._codes import life
from ._groups import free_group


class PairCellElement(Element):
    """Cell of an automaton over the product `H x H` of the integers modulo `n`.

    For a prime `n` the product is `K^2`.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        first:
            The initial first coordinate, an integer from 0 to `modulus - 1`.

        second:
            The initial second coordinate, an integer from 0 to `modulus - 1`.

        modulus:
            The order `n` of the cyclic group `H`, the size of the field `K` when prime.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import PairCellElement
        >>> PairCellElement('cell', first=1).modulus
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        first: Annotated[float, StateSpec(domain=ZZ)] = 0,
        second: Annotated[float, StateSpec(domain=ZZ)] = 0,
        modulus: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.first = first
        self.second = second
        self.modulus = modulus


FREE_GENERATORS = GeneratorNeighborhood(('a', 'b'), center=False)
TREE_RANK = 4
MIN_MODULUS = 2


def _pair_cells(
    space: CayleySpace,
    modulus: int,
    initial: npt.ArrayLike | None,
    random_state: int | np.random.Generator | None,
) -> Array:
    """Return the initial pairs, one row of two coordinates per site.

    By default they are random modulo the modulus.
    """
    if modulus < MIN_MODULUS:
        error_msg = f'The order of the cyclic group H is at least 2, but it is {modulus}.'
        raise ParameterError(error_msg)
    size = space.prepare().size_
    if initial is not None:
        return np.asarray(initial, dtype=float).reshape(size, 2)
    return np.random.default_rng(random_state).integers(0, modulus, (size, 2)).astype(float)


def _pair_system(
    label: str,
    rule: object,
    *,
    radius: int,
    modulus: int,
    initial: npt.ArrayLike | None,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return an automaton of pairs on a ball of the free group of rank 2.

    It reads `a`, `a^-1`, `b`, and `b^-1`.
    """
    space = CayleySpace(free_group(2), ball=radius)
    pairs = _pair_cells(space, modulus, initial, random_state)
    values = {'first': pairs[:, 0], 'second': pairs[:, 1], 'modulus': modulus}
    return System(
        label,
        elements=[LatticeElement('cells', element=PairCellElement, space=space, values=values)],
        interactions=[
            LocalRuleInteraction(
                'rule',
                target='cells',
                state=('first', 'second'),
                rule=rule,
                neighborhood=FREE_GENERATORS,
            ),
        ],
    )


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

    Raises:
        ParameterError:
            The modulus is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import MullerAutomatonSystem
        >>> MullerAutomatonSystem(radius=2).prepare().system_.elements[0].prepare().size_
        17
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'free group', 'Garden of Eden')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1
    observables = (
        ObservableSpec('first_mean', group='cells', variable='first'),
        ObservableSpec('second_mean', group='cells', variable='second'),
    )

    def __init__(
        self,
        *,
        radius: int = 5,
        modulus: int = 2,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.modulus = modulus
        self.initial = initial
        self.random_state = random_state

    @staticmethod
    def _muller(*values: Array, modulus: Array | float) -> tuple[Array, Array]:
        """Return `p1(x(ga)) + p2(x(gb)) + p1(x(ga^-1)) + p2(x(gb^-1))`.

        The projections are `p1(h, k) = (h, 0)` and `p2(h, k) = (k, 0)`.
        """
        first, second = values[:4], values[4:]
        total = first[0] + first[1] + second[2] + second[3]
        return total % modulus, np.zeros_like(total)

    def _build_system(self) -> System:
        """Return Muller's automaton on the free group F2.

        It is pre-injective but not surjective.
        """
        return _pair_system(
            self.label,
            self._muller,
            radius=self.radius,
            modulus=self.modulus,
            initial=self.initial,
            random_state=self.random_state,
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

    Raises:
        ParameterError:
            The modulus is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SurjectiveLinearAutomatonSystem
        >>> SurjectiveLinearAutomatonSystem(radius=2).prepare().system_.elements[0].prepare().size_
        17
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'free group', 'linear', 'Garden of Eden')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1
    observables = (
        ObservableSpec('first_mean', group='cells', variable='first'),
        ObservableSpec('second_mean', group='cells', variable='second'),
    )

    def __init__(
        self,
        *,
        radius: int = 5,
        modulus: int = 2,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.modulus = modulus
        self.initial = initial
        self.random_state = random_state

    @staticmethod
    def _surjective_linear(*values: Array, modulus: Array | float) -> tuple[Array, Array]:
        """Return `q1(x(ga)) + q1(x(ga^-1)) + q2(x(gb)) + q2(x(gb^-1))`.

        The maps are `q1(h, k) = (h, 0)` and `q2(h, k) = (0, h)`.
        """
        first = values[:4]
        return (first[0] + first[1]) % modulus, (first[2] + first[3]) % modulus

    def _build_system(self) -> System:
        """Return a linear automaton over `K^2` on the free group F2.

        It is surjective but not pre-injective.
        """
        return _pair_system(
            self.label,
            self._surjective_linear,
            radius=self.radius,
            modulus=self.modulus,
            initial=self.initial,
            random_state=self.random_state,
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

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LifeInATreeSystem
        >>> LifeInATreeSystem(radius=2).prepare().system_.elements[0].prepare().size_
        65
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'life', 'free group', 'Garden of Eden')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 20
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 4,
        density: float = 0.3,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.density = density
        self.initial = initial
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the Game of Life on the free group F4, surjective but not pre-injective.

        The free group F4 is a tree of degree 8.
        """
        space = CayleySpace(free_group(TREE_RANK), ball=self.radius)
        size = space.prepare().size_
        if self.initial is None:
            cells = (np.random.default_rng(self.random_state).random(size) < self.density).astype(float)
        else:
            cells = np.asarray(self.initial, dtype=float)
        return System(
            self.label,
            elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': cells})],
            interactions=[
                LocalRuleInteraction(
                    'life',
                    target='cells',
                    state='alive',
                    rule=life((3,), (2, 3)),
                    neighborhood=GeneratorNeighborhood(space.generators_, center=False),
                ),
            ],
        )
