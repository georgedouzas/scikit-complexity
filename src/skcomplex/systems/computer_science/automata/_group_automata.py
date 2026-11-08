"""Automata on groups: majority and parity rules, a Laplacian, and the golden mean shift.

The Laplacian is linear. The majority and parity rules run on Cayley balls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import (
    BoxNeighborhood,
    CayleySpace,
    GeneratorNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
)
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, SubshiftSpec, TotalisticSpec, VisualSpec
from ....core.types import Array
from ._binary_cell import BinaryCellElement
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter

GOLDEN_MEAN = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))


class GoldenCellElement(Element):
    """Cell of the golden mean shift, whose state `alive` is 0 or 1.

    Its configurations hold no two adjacent ones.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        alive:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GoldenCellElement
        >>> GoldenCellElement('cell', alive=1).alive
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec(domain=GOLDEN_MEAN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive


class LaplacianCellElement(Element):
    """Cell of an automaton over the field of a prime number of elements.

    An example is the discrete Laplacian on a ring.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The initial value, an integer from 0 to `modulus - 1`.

        modulus:
            The prime size of the field.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LaplacianCellElement
        >>> LaplacianCellElement('cell').modulus
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=ZZ)] = 0,
        modulus: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.modulus = modulus


GENERATOR_MEMORY = MemoryNeighborhood(('1', 'a', 'a^-1'))
GRIGORCHUK_MEMORY = MemoryNeighborhood(('1', 'a', 'b', 'c', 'd'))
LAMPLIGHTER_MEMORY = MemoryNeighborhood(('1', 'a', 'b'))
FLIP_PROBABILITY = 0.5


def _majority(center: Array, *values: Array, degree: Array) -> Array:
    """Return the value held by more than half of a cell and its neighbours.

    Only the neighbours inside the space count, and a tie keeps the value.
    """
    total = center + np.sum(values, axis=0)
    count = degree + 1
    return np.where(2 * total > count, 1.0, np.where(2 * total < count, 0.0, center))


def _random_cells(size: int, density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < density).astype(float)


def _majority_system(
    label: str,
    space: CayleySpace,
    memory: MemoryNeighborhood,
    density: float,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return the majority automaton on a group space from random cells."""
    cells = _random_cells(space.prepare().size_, density, random_state)
    return System(
        label,
        elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': cells})],
        interactions=[
            LocalRuleInteraction('majority', target='cells', state='alive', rule=_majority, neighborhood=memory),
        ],
    )


class FreeGroupAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the free group.

    The ball is a tree in which every cell follows its neighbours. Every cell reads itself and the cells at its
    generators and their inverses, and takes the majority value, so the random start freezes into blocks of agreeing
    cells. Cells on the boundary count only their neighbours inside the ball. The system has the cells `cells.alive`.

    Args:
        rank:
            The number of generators.

        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import FreeGroupAutomatonSystem
        >>> FreeGroupAutomatonSystem(radius=3).prepare().system_.elements[0].prepare().size_
        53
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'free group')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 20
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        rank: int = 2,
        *,
        radius: int = 6,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.rank = rank
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        space = CayleySpace(free_group(self.rank), ball=self.radius)
        return _majority_system(
            self.label,
            space,
            GeneratorNeighborhood(space.prepare().generators_),
            self.density,
            self.random_state,
        )


class LamplighterAutomatonSystem(BaseSystem):
    """The parity automaton on a ball of the lamplighter group.

    It is a linear rule over the field of two elements. Every cell takes the sum modulo 2 of itself and the cells at the
    generators `a` and `b`, so a few ones spread into nested, self-similar patterns over the Diestel-Leader graph of the
    group. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LamplighterAutomatonSystem
        >>> LamplighterAutomatonSystem(radius=3).prepare().system_.elements[0].prepare().size_
        39
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'linear', 'lamplighter')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Grigorchuk and Zuk 2001')
    until = 20
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 7,
        density: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        space = CayleySpace(lamplighter(), ball=self.radius)
        cells = _random_cells(space.prepare().size_, self.density, self.random_state)
        parity = TotalisticSpec({total: total % 2 for total in range(len(LAMPLIGHTER_MEMORY.elements) + 1)})
        return System(
            self.label,
            elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': cells})],
            interactions=[
                LocalRuleInteraction(
                    'parity',
                    target='cells',
                    state='alive',
                    rule=parity,
                    neighborhood=LAMPLIGHTER_MEMORY,
                ),
            ],
        )


class HeisenbergAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the discrete Heisenberg group.

    The group has polynomial growth of degree 4. Every cell reads itself and the cells at `a`, `a^-1`, `b`, and `b^-1`,
    on the right, and takes the majority value. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import HeisenbergAutomatonSystem
        >>> HeisenbergAutomatonSystem(radius=2).prepare().system_.elements[0].prepare().size_
        17
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'nilpotent')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 20
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 5,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        space = CayleySpace(heisenberg(), ball=self.radius)
        return _majority_system(
            self.label,
            space,
            GeneratorNeighborhood(space.prepare().generators_),
            self.density,
            self.random_state,
        )


class BaumslagSolitarAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the Baumslag-Solitar group `BS(m, n)`.

    The group has exponential growth. Every cell reads itself and the cells at `a`, `a^-1`, `t`, and `t^-1`, and takes
    the majority value. The ball is built from a confluent rewriting system of the presentation, which GAP finds for
    `BS(1, n)`. The system has the cells `cells.alive`.

    Args:
        m:
            The exponent of `a` conjugated by `t`.

        n:
            The exponent it is conjugated to.

        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import BaumslagSolitarAutomatonSystem
        >>> BaumslagSolitarAutomatonSystem(radius=2).prepare().system_.elements[0].prepare().size_
        17
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'solvable')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Baumslag and Solitar 1962')
    until = 20
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        m: int = 1,
        n: int = 2,
        *,
        radius: int = 6,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.m = m
        self.n = n
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the majority automaton on a ball of a Baumslag-Solitar group.

        The group is `BS(m, n)`, of exponential growth.
        """
        space = CayleySpace(baumslag_solitar(self.m, self.n), ball=self.radius)
        return _majority_system(
            self.label,
            space,
            GeneratorNeighborhood(space.prepare().generators_),
            self.density,
            self.random_state,
        )


class LinearRingAutomatonSystem(BaseSystem):
    """The discrete Laplacian as a linear automaton on a ring.

    It runs over the field of `modulus` elements.

    The ring is the cyclic group of order `size`, a periodic lattice, and every cell becomes `x(g - 1) + x(g + 1) - 2
    x(g)` modulo the field size, the element `a^-1 + a - 2` of the group ring. From a single 1 in the middle the values
    spread as binomial coefficients reduced modulo the field size, a self-similar triangle. The system has the cells
    `cells.value`.

    Args:
        size:
            The order of the cyclic group, the number of cells of the ring.

        modulus:
            The prime size of the field.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LinearRingAutomatonSystem
        >>> LinearRingAutomatonSystem(11).prepare().system_.elements[0].prepare().size_
        11
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'linear', 'Laplacian')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 40
    dt = 1
    observables = (ObservableSpec('mean', group='cells', variable='value'),)

    def __init__(
        self,
        size: int = 101,
        *,
        modulus: int = 3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.modulus = modulus

    @staticmethod
    def _laplacian(left: Array, center: Array, right: Array, *, modulus: float) -> Array:
        """Return the discrete Laplacian `x(g - 1) + x(g + 1) - 2 x(g)`.

        It is taken modulo the field size.
        """
        return (left + right - 2 * center) % modulus

    def _build_system(self) -> System:
        """Return the discrete Laplacian as a linear automaton on a ring.

        It runs over the field of `modulus` elements.
        """
        values = np.zeros(self.size)
        values[self.size // 2] = 1.0
        return System(
            self.label,
            elements=[
                LatticeElement(
                    'cells',
                    element=LaplacianCellElement,
                    space=LatticeSpace(self.size),
                    values={'value': values, 'modulus': self.modulus},
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'laplacian',
                    target='cells',
                    state='value',
                    rule=self._laplacian,
                    neighborhood=BoxNeighborhood(),
                ),
            ],
        )


class SubshiftsSystem(BaseSystem):
    """A Markov chain on the golden mean shift.

    The shift holds the configurations of a ring that never hold two adjacent ones.

    The ring is the cyclic group of order `size`, a periodic lattice. It starts empty, and every step proposes to flip
    every cell in a random order, rejecting the flips that would put two ones side by side. The chain samples the
    admissible configurations, whose number grows as the power of the golden ratio. The system has the cells
    `cells.alive`.

    Args:
        size:
            The number of cells of the ring, the order of the cyclic group.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SubshiftsSystem
        >>> SubshiftsSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'computer_science/automata'
    topics = ('group', 'subshift', 'golden mean', 'Markov chain')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Lind and Marcus 1995')
    until = 50
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        size: int = 200,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size

    @staticmethod
    def _golden_move(left: Array, center: Array, right: Array, *, random: Array) -> Array:
        """Return the cell flipped with probability one half.

        The flip is refused when a 1 would sit next to another 1.
        """
        flipped = np.where(random < FLIP_PROBABILITY, 1.0 - center, center)
        blocked = (flipped == 1) & ((left == 1) | (right == 1))
        return np.where(blocked, center, flipped)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        flips = LocalRuleInteraction(
            'flips',
            target='cells',
            state='alive',
            rule=self._golden_move,
            neighborhood=BoxNeighborhood(),
            schedule='random-sequential',
            stochastic=True,
        )
        return System(
            self.label,
            elements=[LatticeElement('cells', element=GoldenCellElement, space=LatticeSpace(self.size))],
            interactions=[flips],
        )


class GrigorchukAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the Grigorchuk group.

    The group has intermediate growth. Every cell reads itself and the cells at the generators `a`, `b`, `c`, and `d`,
    all of order 2, and takes the majority value. The ball of radius 8 holds 271 sites, and cells on its boundary count
    only their neighbours inside it. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GrigorchukAutomatonSystem
        >>> GrigorchukAutomatonSystem(radius=3).prepare().system_.elements[0].prepare().size_
        23
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'intermediate growth')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Grigorchuk 1984')
    until = 20
    dt = 1
    observables = (ObservableSpec('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 8,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the majority automaton on a ball of the Grigorchuk group.

        The group has intermediate growth.
        """
        return _majority_system(
            self.label,
            CayleySpace(grigorchuk_group(), ball=self.radius),
            GRIGORCHUK_MEMORY,
            self.density,
            self.random_state,
        )
