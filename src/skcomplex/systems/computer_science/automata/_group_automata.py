"""Automata on groups: majority and parity rules on Cayley balls, a linear Laplacian, and the golden mean shift."""

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
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, SubshiftSpec, TotalisticSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ._binary_cell import BinaryCellElement
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter

GOLDEN_MEAN = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))


class GoldenCellElement(Element):
    """Cell of the golden mean shift, whose state `alive` is 0 or 1 and whose configurations hold no two adjacent ones.

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
    """Cell of an automaton over the field of a prime number of elements, such as the discrete Laplacian on a ring.

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
    """Return the value held by more than half of a cell and its neighbours inside the space, keeping it on a tie."""
    total = center + np.sum(values, axis=0)
    count = degree + 1
    return np.where(2 * total > count, 1.0, np.where(2 * total < count, 0.0, center))


def _random_cells(size: int, density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < density).astype(float)


def _density_observables(system: System) -> list[ObservableSpec]:
    """Return the density of ones of the cells."""
    return [ObservableSpec('density', group='cells', variable='alive')]


def _mean_observables(system: System) -> list[ObservableSpec]:
    """Return the mean value of the cells."""
    return [ObservableSpec('mean', group='cells', variable='value')]


def _laplacian(left: Array, center: Array, right: Array, *, modulus: float) -> Array:
    """Return the discrete Laplacian `x(g - 1) + x(g + 1) - 2 x(g)` modulo the field size."""
    return (left + right - 2 * center) % modulus


def _golden_move(left: Array, center: Array, right: Array, *, random: Array) -> Array:
    """Return the cell flipped with probability one half, unless a 1 would sit next to another 1."""
    flipped = np.where(random < FLIP_PROBABILITY, 1.0 - center, center)
    blocked = (flipped == 1) & ((left == 1) | (right == 1))
    return np.where(blocked, center, flipped)


def _majority_system(
    space: CayleySpace,
    memory: MemoryNeighborhood,
    density: float,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return the majority automaton on a group space from random cells."""
    cells = _random_cells(space.prepare().size_, density, random_state)
    return System(
        elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': cells})],
        interactions=[
            LocalRuleInteraction('majority', target='cells', state='alive', rule=_majority, neighborhood=memory),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'majority', 'free group'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 20, 'dt': 1},
    observables=_density_observables,
)
def free_group_automaton(
    rank: int = 2,
    *,
    radius: int = 6,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the majority automaton on a ball of the free group, a tree in which every cell follows its neighbours.

    Every cell reads itself and the cells at its generators and their inverses, and takes the majority value, so the
    random start freezes into blocks of agreeing cells. Cells on the boundary count only their neighbours inside the
    ball.

    Args:
        rank:
            The number of generators.

        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import free_group_automaton
        >>> free_group_automaton(radius=3).elements[0].prepare().size_
        53
    """
    space = CayleySpace(free_group(rank), ball=radius)
    return _majority_system(
        space,
        GeneratorNeighborhood(space.prepare().generators_),
        density,
        random_state,
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'linear', 'lamplighter'),
    references=('Ceccherini-Silberstein and Coornaert 2023', 'Grigorchuk and Zuk 2001'),
    defaults={'until': 20, 'dt': 1},
    observables=_density_observables,
)
def lamplighter_automaton(
    *,
    radius: int = 7,
    density: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the parity automaton on a ball of the lamplighter group, a linear rule over the field of two elements.

    Every cell takes the sum modulo 2 of itself and the cells at the generators `a` and `b`, so a few ones spread into
    nested, self-similar patterns over the Diestel-Leader graph of the group.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import lamplighter_automaton
        >>> lamplighter_automaton(radius=3).elements[0].prepare().size_
        39
    """
    space = CayleySpace(lamplighter(), ball=radius)
    cells = _random_cells(space.prepare().size_, density, random_state)
    parity = TotalisticSpec({total: total % 2 for total in range(len(LAMPLIGHTER_MEMORY.elements) + 1)})
    return System(
        elements=[LatticeElement('cells', element=BinaryCellElement, space=space, values={'alive': cells})],
        interactions=[
            LocalRuleInteraction('parity', target='cells', state='alive', rule=parity, neighborhood=LAMPLIGHTER_MEMORY),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'majority', 'nilpotent'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 20, 'dt': 1},
    observables=_density_observables,
)
def heisenberg_automaton(
    *,
    radius: int = 5,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the majority automaton on a ball of the discrete Heisenberg group, of polynomial growth of degree 4.

    Every cell reads itself and the cells at `a`, `a^-1`, `b`, and `b^-1`, on the right, and takes the majority value.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import heisenberg_automaton
        >>> heisenberg_automaton(radius=2).elements[0].prepare().size_
        17
    """
    space = CayleySpace(heisenberg(), ball=radius)
    return _majority_system(
        space,
        GeneratorNeighborhood(space.prepare().generators_),
        density,
        random_state,
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'majority', 'solvable'),
    references=('Ceccherini-Silberstein and Coornaert 2023', 'Baumslag and Solitar 1962'),
    defaults={'until': 20, 'dt': 1},
    observables=_density_observables,
)
def baumslag_solitar_automaton(
    m: int = 1,
    n: int = 2,
    *,
    radius: int = 6,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the majority automaton on a ball of the Baumslag-Solitar group `BS(m, n)`, of exponential growth.

    Every cell reads itself and the cells at `a`, `a^-1`, `t`, and `t^-1`, and takes the majority value. The ball is
    built from a confluent rewriting system of the presentation, which GAP finds for `BS(1, n)`.

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

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import baumslag_solitar_automaton
        >>> baumslag_solitar_automaton(radius=2).elements[0].prepare().size_
        17
    """
    space = CayleySpace(baumslag_solitar(m, n), ball=radius)
    return _majority_system(
        space,
        GeneratorNeighborhood(space.prepare().generators_),
        density,
        random_state,
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'linear', 'Laplacian'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 40, 'dt': 1},
    observables=_mean_observables,
)
def linear_ring_automaton(
    size: int = 101,
    *,
    modulus: int = 3,
) -> System:
    """Return the discrete Laplacian as a linear automaton on a ring, over the field of `modulus` elements.

    The ring is the cyclic group of order `size`, a periodic lattice, and every cell becomes
    `x(g - 1) + x(g + 1) - 2 x(g)` modulo the field size, the element `a^-1 + a - 2` of the group ring. From a single
    1 in the middle the values spread as binomial coefficients reduced modulo the field size, a self-similar triangle.

    Args:
        size:
            The order of the cyclic group, the number of cells of the ring.

        modulus:
            The prime size of the field.

    Returns:
        system:
            The system, with the cells `cells.value`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import linear_ring_automaton
        >>> linear_ring_automaton(11).elements[0].prepare().size_
        11
    """
    values = np.zeros(size)
    values[size // 2] = 1.0
    return System(
        elements=[
            LatticeElement(
                'cells',
                element=LaplacianCellElement,
                space=LatticeSpace(size),
                values={'value': values, 'modulus': modulus},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'laplacian',
                target='cells',
                state='value',
                rule=_laplacian,
                neighborhood=BoxNeighborhood(),
            ),
        ],
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'subshift', 'golden mean', 'Markov chain'),
    references=('Ceccherini-Silberstein and Coornaert 2023', 'Lind and Marcus 1995'),
    defaults={'until': 50, 'dt': 1},
    observables=_density_observables,
)
def subshifts(size: int = 200) -> System:
    """Return a Markov chain on the golden mean shift, the configurations of a ring that never hold two adjacent ones.

    The ring is the cyclic group of order `size`, a periodic lattice. It starts empty, and every step proposes to flip
    every cell in a random order, rejecting the flips that would put two ones side by side. The chain samples the
    admissible configurations, whose number grows as the power of the golden ratio.

    Args:
        size:
            The number of cells of the ring, the order of the cyclic group.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import subshifts
        >>> subshifts(10).elements[0].prepare().size_
        10
    """
    flips = LocalRuleInteraction(
        'flips',
        target='cells',
        state='alive',
        rule=_golden_move,
        neighborhood=BoxNeighborhood(),
        schedule='random-sequential',
        stochastic=True,
    )
    return System(
        elements=[LatticeElement('cells', element=GoldenCellElement, space=LatticeSpace(size))],
        interactions=[flips],
    )


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'majority', 'intermediate growth'),
    references=('Ceccherini-Silberstein and Coornaert 2023', 'Grigorchuk 1984'),
    defaults={'until': 20, 'dt': 1},
    observables=_density_observables,
)
def grigorchuk_automaton(
    *,
    radius: int = 8,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the majority automaton on a ball of the Grigorchuk group, a group of intermediate growth.

    Every cell reads itself and the cells at the generators `a`, `b`, `c`, and `d`, all of order 2, and takes the
    majority value. The ball of radius 8 holds 271 sites, and cells on its boundary count only their neighbours
    inside it.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import grigorchuk_automaton
        >>> grigorchuk_automaton(radius=3).elements[0].prepare().size_
        23
    """
    return _majority_system(CayleySpace(grigorchuk_group(), ball=radius), GRIGORCHUK_MEMORY, density, random_state)
