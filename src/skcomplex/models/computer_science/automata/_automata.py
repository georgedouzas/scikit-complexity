"""Automata models: elementary rules, life, and turmites."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, state
from ....core.components import Element
from ....core.laws import Law, LocalRule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace, Neighborhood
from ....core.system import System
from ._codes import Wolfram, life

type Array = npt.NDArray[np.float64]


class ElementaryCell(Element):
    """Cell of a one-dimensional automaton that follows an elementary rule by its Wolfram number.

    Args:
        label:
            The label of the cell or the lattice group.

        alive:
            The initial value, 0 or 1.

        rule:
            The Wolfram number of the rule.

    Examples:
        >>> from skcomplex.models.computer_science.automata import ElementaryCell
        >>> ElementaryCell('cell', rule=110).rule
        110
    """

    alive: float = state(default=0, domain=Domain.BOOLEAN)
    rule: int = 30

    def laws(self, s: Symbols) -> list[Law]:
        """Return the elementary rule on the left, centre, and right cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.alive, Neighborhood.radius(), Wolfram(self.rule))]


class LifeCell(Element):
    """Cell of a life-like automaton, born and surviving by its number of live Moore neighbours.

    Args:
        label:
            The label of the cell or the lattice group.

        alive:
            The initial value, 0 or 1.

        born:
            The numbers of live neighbours that bring a dead cell to life.

        survive:
            The numbers of live neighbours that keep a live cell alive.

    Examples:
        >>> from skcomplex.models.computer_science.automata import LifeCell
        >>> LifeCell('cell').survive
        (2, 3)
    """

    alive: float = state(default=0, domain=Domain.BOOLEAN)
    born: tuple[int, ...] = (3,)
    survive: tuple[int, ...] = (2, 3)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the outer totalistic rule of the cell.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.alive, Neighborhood.moore(), life(self.born, self.survive))]


def _ant_rule(*values: Array) -> tuple[Array, Array]:
    """Return the next colors and ants of Langton's ant from the von Neumann neighbourhoods of both states."""
    colors, ants = values[:5], values[5:]
    color = np.where(ants[0] >= 0, 1 - colors[0], colors[0])
    ant = np.full(color.shape, -1.0)
    arrivals = {1: 2, 2: 1, 3: 3, 4: 0}
    for column, heading in arrivals.items():
        turned = np.where(colors[column] == 0, (ants[column] + 1) % 4, (ants[column] + 3) % 4)
        ant = np.where((ants[column] >= 0) & (turned == heading), turned, ant)
    return color, ant


class AntCell(Element):
    """Cell of Langton's ant: a color, and the heading of the ant on it or -1.

    The ant turns right on a white cell and left on a black one, flips the cell's color, and moves one cell forward.

    Args:
        label:
            The label of the cell or the lattice group.

        color:
            The initial color, 0 for white and 1 for black.

        ant:
            The initial heading of an ant on the cell, 0 to 3 for north, east, south, and west, or -1 for none.

    Examples:
        >>> from skcomplex.models.computer_science.automata import AntCell
        >>> AntCell('cell').ant
        -1
    """

    color: float = state(default=0, domain=Domain.BOOLEAN)
    ant: float = state(default=-1, domain=Domain.INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that moves the ant and flips colors.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRule((s.color, s.ant), Neighborhood.von_neumann(center=True), _ant_rule, reads=(s.color, s.ant)),
        ]


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'elementary', 'chaos'),
    references=('Wolfram 1983', 'Charbonneau 2017'),
    defaults={'until': 100, 'dt': 1},
)
def elementary_automaton(rule: int = 30, *, size: int = 101, initial: Sequence[float] | None = None) -> System:
    """Return an elementary cellular automaton by its Wolfram number, started from a single live cell by default.

    Args:
        rule:
            The Wolfram number from 0 to 255.

        size:
            The number of cells of the ring.

        initial:
            The initial cells, or `None` for a single live cell in the middle.

    Returns:
        system:
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import elementary_automaton
        >>> elementary_automaton(90, size=11).elements[0].size
        11
    """
    values = np.zeros(size) if initial is None else np.asarray(initial, dtype=float)
    if initial is None:
        values[size // 2] = 1.0
    return System(components=[ElementaryCell.lattice('cells', LatticeSpace(size), alive=values, rule=rule)])


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'life', 'emergence'),
    references=('Gardner 1970', 'Downey 2018'),
    defaults={'until': 100, 'dt': 1},
)
def game_of_life(
    shape: Sequence[int] = (100, 100),
    *,
    density: float = 0.3,
    initial: npt.ArrayLike | None = None,
    born: Sequence[int] = (3,),
    survive: Sequence[int] = (2, 3),
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Conway's Game of Life, or another life-like rule, on a torus.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a cell starts alive, when no initial cells are given.

        initial:
            The initial cells, an array of the shape, or `None` for random cells.

        born:
            The numbers of live neighbours that bring a dead cell to life.

        survive:
            The numbers of live neighbours that keep a live cell alive.

        random_state:
            The seed or generator of the random initial cells.

    Returns:
        system:
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import game_of_life
        >>> game_of_life((20, 30)).elements[0].size
        600
    """
    values = _random_cells(shape, density, random_state) if initial is None else np.asarray(initial, dtype=float)
    space = LatticeSpace(tuple(shape))
    return System(
        components=[LifeCell.lattice('cells', space, alive=values, born=tuple(born), survive=tuple(survive))],
    )


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'ant', 'emergence'),
    references=('Langton 1986',),
    defaults={'until': 11000, 'dt': 100},
)
def langtons_ant(shape: Sequence[int] = (80, 80), *, heading: int = 0) -> System:
    """Return Langton's ant, which wanders chaotically for about 10,000 steps before it builds a periodic highway.

    Args:
        shape:
            The number of rows and columns of the torus.

        heading:
            The initial heading of the ant, 0 to 3 for north, east, south, and west.

    Returns:
        system:
            The model, with the colors `cells.color` and the ant `cells.ant`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import langtons_ant
        >>> int((langtons_ant((9, 9)).elements[0].values['ant'] >= 0).sum())
        1
    """
    ant = np.full(tuple(shape), -1.0)
    ant[shape[0] // 2, shape[1] // 2] = heading
    return System(components=[AntCell.lattice('cells', LatticeSpace(tuple(shape)), ant=ant)])
