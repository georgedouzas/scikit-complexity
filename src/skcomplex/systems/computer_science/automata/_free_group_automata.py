"""Automata on free groups beyond the Garden of Eden theorem: Muller's automaton, a linear one, and Life in a tree."""

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
from ....core.components.system import System
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ._binary_cell import BinaryCellElement
from ._codes import life
from ._groups import free_group


class PairCellElement(Element):
    """Cell of an automaton over the product `H x H` of the integers modulo `n` with itself, or `K^2` for a prime `n`.

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


def _muller(*values: Array, modulus: Array | float) -> tuple[Array, Array]:
    """Return `p1(x(ga)) + p2(x(gb)) + p1(x(ga^-1)) + p2(x(gb^-1))` for `p1(h, k) = (h, 0)` and `p2(h, k) = (k, 0)`."""
    first, second = values[:4], values[4:]
    total = first[0] + first[1] + second[2] + second[3]
    return total % modulus, np.zeros_like(total)


def _surjective_linear(*values: Array, modulus: Array | float) -> tuple[Array, Array]:
    """Return `q1(x(ga)) + q1(x(ga^-1)) + q2(x(gb)) + q2(x(gb^-1))` for `q1(h, k) = (h, 0)` and `q2(h, k) = (0, h)`."""
    first = values[:4]
    return (first[0] + first[1]) % modulus, (first[2] + first[3]) % modulus


def _pair_cells(
    space: CayleySpace,
    modulus: int,
    initial: npt.ArrayLike | None,
    random_state: int | np.random.Generator | None,
) -> Array:
    """Return the initial pairs, one row of two coordinates per site, random modulo the modulus by default."""
    if modulus < MIN_MODULUS:
        error_msg = f'The order of the cyclic group H is at least 2, but it is {modulus}.'
        raise ParameterError(error_msg)
    size = space.prepare().size_
    if initial is not None:
        return np.asarray(initial, dtype=float).reshape(size, 2)
    return np.random.default_rng(random_state).integers(0, modulus, (size, 2)).astype(float)


def _pair_system(
    rule: object,
    *,
    radius: int,
    modulus: int,
    initial: npt.ArrayLike | None,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return an automaton of pairs on a ball of the free group of rank 2 that reads `a`, `a^-1`, `b`, and `b^-1`."""
    space = CayleySpace(free_group(2), ball=radius)
    pairs = _pair_cells(space, modulus, initial, random_state)
    values = {'first': pairs[:, 0], 'second': pairs[:, 1], 'modulus': modulus}
    return System(
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


def _pair_observables(system: System) -> list[ObservableSpec]:
    """Return the means of both coordinates of the cells."""
    return [
        ObservableSpec('first_mean', group='cells', variable='first'),
        ObservableSpec('second_mean', group='cells', variable='second'),
    ]


def _density_observables(system: System) -> list[ObservableSpec]:
    """Return the density of ones of the cells."""
    return [ObservableSpec('density', group='cells', variable='alive')]


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'free group', 'Garden of Eden'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 10, 'dt': 1},
    observables=_pair_observables,
)
def muller_automaton(
    *,
    radius: int = 5,
    modulus: int = 2,
    initial: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Muller's automaton on the free group F2, pre-injective but not surjective.

    The alphabet is `A = H x H` for the cyclic group `H` of order `modulus`, and every cell becomes
    `p1(x(ga)) + p2(x(gb)) + p1(x(ga^-1)) + p2(x(gb^-1))`, where `p1(h, k) = (h, 0)` and `p2(h, k) = (k, 0)`
    (Ceccherini-Silberstein and Coornaert 2023, Section 5.11). The image lies in `H x {0}`, so the automaton is not
    surjective, yet two configurations that differ on a finite set have different images, so it is pre-injective, which
    the Garden of Eden theorem forbids on amenable groups. For a prime modulus it is the linear automaton over `K^2` of
    Section 8.10. Cells on the boundary of the ball read 0 outside it.

    Args:
        radius:
            The radius of the ball of F2.

        modulus:
            The order of `H`, at least 2.

        initial:
            The initial pairs, an array of shape `(sites, 2)`, or `None` for random ones.

        random_state:
            The seed or generator of the random initial pairs.

    Returns:
        system:
            The system, with the coordinates `cells.first` and `cells.second`.

    Raises:
        ParameterError:
            The modulus is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import muller_automaton
        >>> muller_automaton(radius=2).elements[0].prepare().size_
        17
    """
    return _pair_system(_muller, radius=radius, modulus=modulus, initial=initial, random_state=random_state)


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'free group', 'linear', 'Garden of Eden'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 10, 'dt': 1},
    observables=_pair_observables,
)
def surjective_linear_automaton(
    *,
    radius: int = 5,
    modulus: int = 2,
    initial: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a linear automaton over `K^2` on the free group F2, surjective but not pre-injective.

    Every cell becomes `q1(x(ga)) + q1(x(ga^-1)) + q2(x(gb)) + q2(x(gb^-1))`, where `q1(h, k) = (h, 0)` and
    `q2(h, k) = (0, h)`, over the field of a prime number `modulus` of elements (Ceccherini-Silberstein and Coornaert
    2023, Section 8.11). The configuration `(0, 1)` at the identity and 0 elsewhere maps to 0, so the automaton is not
    pre-injective, while a preimage of every configuration is built by induction on the word length, so it is
    surjective, which the Garden of Eden theorem forbids on amenable groups. Cells on the boundary of the ball read 0
    outside it.

    Args:
        radius:
            The radius of the ball of F2.

        modulus:
            The prime size of the field `K`.

        initial:
            The initial pairs, an array of shape `(sites, 2)`, or `None` for random ones.

        random_state:
            The seed or generator of the random initial pairs.

    Returns:
        system:
            The system, with the coordinates `cells.first` and `cells.second`.

    Raises:
        ParameterError:
            The modulus is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import surjective_linear_automaton
        >>> surjective_linear_automaton(radius=2).elements[0].prepare().size_
        17
    """
    return _pair_system(_surjective_linear, radius=radius, modulus=modulus, initial=initial, random_state=random_state)


@register_system(
    field='computer_science/automata',
    tags=('group', 'cellular automaton', 'life', 'free group', 'Garden of Eden'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 20, 'dt': 1},
    observables=_density_observables,
)
def life_in_a_tree(
    *,
    radius: int = 4,
    density: float = 0.3,
    initial: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Game of Life on the free group F4, a tree of degree 8, surjective but not pre-injective.

    The memory set is the identity and the generators `a`, `b`, `c`, `d` with their inverses, and a cell is alive after
    a step when the sum over the memory set is 3, or 4 with the cell alive, Conway's rule on the Cayley graph of F4
    (Ceccherini-Silberstein and Coornaert 2023, Exercise 5.14). On the tree the automaton is surjective but not
    pre-injective, unlike Life on the plane, which is neither. Cells on the boundary of the ball read 0 outside it.

    Args:
        radius:
            The radius of the ball of F4.

        density:
            The probability that a cell starts alive, when no initial cells are given.

        initial:
            The initial cells, one per site, or `None` for random ones.

        random_state:
            The seed or generator of the random initial cells.

    Returns:
        system:
            The system, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import life_in_a_tree
        >>> life_in_a_tree(radius=2).elements[0].prepare().size_
        65
    """
    space = CayleySpace(free_group(TREE_RANK), ball=radius)
    size = space.prepare().size_
    if initial is None:
        cells = (np.random.default_rng(random_state).random(size) < density).astype(float)
    else:
        cells = np.asarray(initial, dtype=float)
    return System(
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
