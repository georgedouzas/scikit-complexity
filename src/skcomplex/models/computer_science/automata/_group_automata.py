"""Automata on groups: majority and parity rules on Cayley balls, a linear Laplacian, and the golden mean shift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, LocalRule, Symbols, Totalistic
from ....core.runtime import register_model
from ....core.space import GroupSpace, LatticeSpace, Memory, Neighborhood
from ....core.system import Observable, System
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter

type Array = npt.NDArray[np.float64]

GRIGORCHUK_MEMORY = Memory(('1', 'a', 'b', 'c', 'd'))
LAMPLIGHTER_MEMORY = Memory(('1', 'a', 'b'))


def _majority(center: Array, *values: Array, degree: Array) -> Array:
    """Return the value held by more than half of a cell and its neighbours inside the space, keeping it on a tie."""
    total = center + np.sum(values, axis=0)
    count = degree + 1
    return np.where(2 * total > count, 1.0, np.where(2 * total < count, 0.0, center))


def _random_cells(size: int, density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(size) < density).astype(float)


def _density_observables(system: System) -> list[Observable]:
    """Return the density of ones of the cells."""
    return [Observable('density', group='cells', variable='alive')]


def _mean_observables(system: System) -> list[Observable]:
    """Return the mean value of the cells."""
    return [Observable('mean', group='cells', variable='value')]


class MajorityCell(Element):
    """Cell of an automaton on a group that takes the value held by the majority of its memory set.

    The memory set starts with the identity, the cell itself. Cells on the boundary of a ball count only their
    neighbours inside it, and a cell keeps its value on a tie.

    Args:
        label:
            The label of the cell or the lattice group.

        alive:
            The initial value, 0 or 1.

        memory:
            The memory set, of odd size, usually the identity and the generators.

    Examples:
        >>> from skcomplex.models.computer_science.automata import MajorityCell
        >>> MajorityCell('cell').memory.elements
        ('1', 'a', 'a^-1')
    """

    alive: float = state(default=0, domain=Domain.BOOLEAN)
    memory: Memory = Memory(('1', 'a', 'a^-1'))

    def laws(self, s: Symbols) -> list[Law]:
        """Return the majority rule on the memory set.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.alive, self.memory, _majority)]


class ParityCell(Element):
    """Cell of an automaton on a group that takes the sum modulo 2 of its memory set, a linear rule over `GF(2)`.

    Args:
        label:
            The label of the cell or the lattice group.

        alive:
            The initial value, 0 or 1.

        memory:
            The memory set.

    Examples:
        >>> from skcomplex.models.computer_science.automata import ParityCell
        >>> len(ParityCell('cell').memory)
        3
    """

    alive: float = state(default=0, domain=Domain.BOOLEAN)
    memory: Memory = LAMPLIGHTER_MEMORY

    def laws(self, s: Symbols) -> list[Law]:
        """Return the parity rule on the memory set.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRule(s.alive, self.memory, Totalistic({total: total % 2 for total in range(len(self.memory) + 1)})),
        ]


def _laplacian(left: Array, center: Array, right: Array, *, modulus: float) -> Array:
    """Return the discrete Laplacian `x(g - 1) + x(g + 1) - 2 x(g)` modulo the field size."""
    return (left + right - 2 * center) % modulus


class LaplacianCell(Element):
    """Cell of the discrete Laplacian on the cyclic group of a ring over the field of a prime number of elements.

    Args:
        label:
            The label of the cell or the lattice group.

        value:
            The initial value, an integer from 0 to `modulus - 1`.

        modulus:
            The prime size of the field.

    Examples:
        >>> from skcomplex.models.computer_science.automata import LaplacianCell
        >>> LaplacianCell('cell').modulus
        3
    """

    value: float = state(default=0, domain=Domain.INTEGER)
    modulus: float = parameter(3, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the linear rule of the Laplacian.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.value, Neighborhood.radius(), _laplacian)]


def _golden_move(left: Array, center: Array, right: Array, *, random: Array) -> Array:
    """Return the cell flipped with probability one half, unless a 1 would sit next to another 1."""
    flipped = np.where(random < 0.5, 1.0 - center, center)  # noqa: PLR2004
    blocked = (flipped == 1) & ((left == 1) | (right == 1))
    return np.where(blocked, center, flipped)


class GoldenMeanCell(Element):
    """Cell of a Markov chain on the golden mean shift, the configurations of a ring without two adjacent ones.

    Every step visits the cells in a random order, and each proposes to flip, which it does unless a one would sit
    next to another one, so the configuration stays admissible.

    Args:
        label:
            The label of the cell or the lattice group.

        alive:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.models.computer_science.automata import GoldenMeanCell
        >>> GoldenMeanCell('cell').alive
        0
    """

    alive: float = state(default=0, domain=Domain.BOOLEAN)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the constrained flip of the cell.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.alive, Neighborhood.radius(), _golden_move, schedule='random-sequential', stochastic=True)]


def _majority_system(
    space: GroupSpace,
    memory: Memory,
    density: float,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return the majority automaton on a group space from random cells."""
    cells = _random_cells(space.size, density, random_state)
    return System(components=[MajorityCell.lattice('cells', space, alive=cells, memory=memory)])


@register_model(
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
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import free_group_automaton
        >>> free_group_automaton(radius=3).elements[0].size
        53
    """
    group = free_group(rank)
    return _majority_system(GroupSpace(group, ball=radius), Memory.generators(group), density, random_state)


@register_model(
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
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import lamplighter_automaton
        >>> lamplighter_automaton(radius=3).elements[0].size
        39
    """
    space = GroupSpace(lamplighter(), ball=radius)
    cells = _random_cells(space.size, density, random_state)
    return System(components=[ParityCell.lattice('cells', space, alive=cells, memory=LAMPLIGHTER_MEMORY)])


@register_model(
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
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import heisenberg_automaton
        >>> heisenberg_automaton(radius=2).elements[0].size
        17
    """
    group = heisenberg()
    return _majority_system(GroupSpace(group, ball=radius), Memory.generators(group), density, random_state)


@register_model(
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
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import baumslag_solitar_automaton
        >>> baumslag_solitar_automaton(radius=2).elements[0].size
        17
    """
    group = baumslag_solitar(m, n)
    return _majority_system(GroupSpace(group, ball=radius), Memory.generators(group), density, random_state)


@register_model(
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
            The model, with the cells `cells.value`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import linear_ring_automaton
        >>> linear_ring_automaton(11).elements[0].size
        11
    """
    values = np.zeros(size)
    values[size // 2] = 1.0
    return System(components=[LaplacianCell.lattice('cells', LatticeSpace(size), value=values, modulus=modulus)])


@register_model(
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
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import subshifts
        >>> subshifts(10).elements[0].size
        10
    """
    return System(components=[GoldenMeanCell.lattice('cells', LatticeSpace(size))])


@register_model(
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
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import grigorchuk_automaton
        >>> grigorchuk_automaton(radius=3).elements[0].size
        23
    """
    return _majority_system(GroupSpace(grigorchuk_group(), ball=radius), GRIGORCHUK_MEMORY, density, random_state)
