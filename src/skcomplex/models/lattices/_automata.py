"""Model cellular automata: elementary rules, life, voting, Langton's ant, traffic, and the hodgepodge machine."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core.base import Domain, parameter, state
from ...core.model import (
    Element,
    LatticeSpace,
    Law,
    LocalRule,
    Neighborhood,
    OuterTotalistic,
    Symbols,
    System,
    Totalistic,
    Wolfram,
)
from ...core.runtime import register_model

type Array = npt.NDArray[np.float64]

MAJORITY = 5


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
        >>> from skcomplex.models.lattices import ElementaryCell
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
        >>> from skcomplex.models.lattices import LifeCell
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
        return [LocalRule(s.alive, Neighborhood.moore(), OuterTotalistic.life(self.born, self.survive))]


class Voter(Element):
    """Cell that takes the majority opinion of its Moore neighbourhood, itself included.

    Args:
        label:
            The label of the cell or the lattice group.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.lattices import Voter
        >>> Voter('cell', opinion=1).opinion
        1
    """

    opinion: float = state(default=0, domain=Domain.BOOLEAN)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the majority rule over the nine cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRule(s.opinion, Neighborhood.moore(center=True), Totalistic(lambda total: total >= MAJORITY)),
        ]


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
        >>> from skcomplex.models.lattices import AntCell
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


class Road(Element):
    """Cell of the Nagel-Schreckenberg road: the velocity of the car on it, or -1 when it is empty.

    Every step, cars accelerate by one up to `vmax`, brake to the gap ahead, slow down by one with probability
    `slowdown`, and move forward by their velocity.

    Args:
        label:
            The label of the cell or the lattice group.

        velocity:
            The initial velocity, or -1 for an empty cell.

        slowdown:
            The probability of random braking.

        vmax:
            The largest velocity.

    Examples:
        >>> from skcomplex.models.lattices import Road
        >>> Road('cell').vmax
        5
    """

    velocity: float = state(default=-1, domain=Domain.INTEGER)
    slowdown: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    vmax: int = 5

    def laws(self, s: Symbols) -> list[Law]:
        """Return the velocity update and the movement of cars.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        vmax = self.vmax

        def accelerate(velocity: Array, *ahead: Array, slowdown: Any, random: Array) -> Array:  # noqa: ANN401
            gap = np.full(velocity.shape, float(vmax))
            for distance in range(vmax, 0, -1):
                gap = np.where(ahead[distance - 1] >= 0, distance - 1, gap)
            following = np.minimum(np.minimum(velocity + 1, vmax), gap)
            following = np.where(random < slowdown, np.maximum(following - 1, 0), following)
            return np.where(velocity >= 0, following, -1.0)

        def move(*behind: Array) -> Array:
            following = np.full(behind[0].shape, -1.0)
            for distance, velocity in enumerate(behind):
                following = np.where(velocity == distance, velocity, following)
            return following

        ahead = Neighborhood(tuple((distance,) for distance in range(vmax + 1)))
        behind = Neighborhood(tuple((-distance,) for distance in range(vmax + 1)))
        return [LocalRule(s.velocity, ahead, accelerate, stochastic=True), LocalRule(s.velocity, behind, move)]


class HodgepodgeCell(Element):
    """Cell of the hodgepodge machine of Gerhardt and Schuster, healthy at 0, infected below `n`, and ill at `n`.

    Healthy cells catch the infection from their infected and ill neighbours, infected cells worsen by the mean of their
    neighbourhood plus `g`, and ill cells heal.

    Args:
        label:
            The label of the cell or the lattice group.

        level:
            The initial level of infection from 0 to `n`.

        n:
            The level of illness.

        k1:
            The weight of infected neighbours for healthy cells.

        k2:
            The weight of ill neighbours for healthy cells.

        g:
            The speed of the infection.

    Examples:
        >>> from skcomplex.models.lattices import HodgepodgeCell
        >>> HodgepodgeCell('cell').n
        100.0
    """

    level: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    n: float = parameter(default=100.0, domain=Domain.POSITIVE)
    k1: float = parameter(default=2.0, domain=Domain.POSITIVE)
    k2: float = parameter(default=3.0, domain=Domain.POSITIVE)
    g: float = parameter(default=28.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of infection, worsening, and healing.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """

        def rule(level: Array, *around: Array, n: Any, k1: Any, k2: Any, g: Any) -> Array:  # noqa: ANN401
            neighbours = np.stack(around)
            infected = ((neighbours > 0) & (neighbours < n)).sum(axis=0)
            ill = (neighbours >= n).sum(axis=0)
            healthy = np.floor(infected / k1) + np.floor(ill / k2)
            worsening = np.minimum(np.floor((level + neighbours.sum(axis=0)) / (infected + ill + 1)) + g, n)
            return np.where(level <= 0, np.minimum(healthy, n), np.where(level >= n, 0.0, worsening))

        return [LocalRule(s.level, Neighborhood.moore(center=True), rule)]


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='cellular automata',
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
        >>> from skcomplex.models.lattices import elementary_automaton
        >>> elementary_automaton(90, size=11).elements[0].size
        11
    """
    values = np.zeros(size) if initial is None else np.asarray(initial, dtype=float)
    if initial is None:
        values[size // 2] = 1.0
    return System(elements=[ElementaryCell.lattice('cells', LatticeSpace(size), alive=values, rule=rule)])


@register_model(
    field='cellular automata',
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
        >>> from skcomplex.models.lattices import game_of_life
        >>> game_of_life((20, 30)).elements[0].size
        600
    """
    values = _random_cells(shape, density, random_state) if initial is None else np.asarray(initial, dtype=float)
    space = LatticeSpace(tuple(shape))
    return System(
        elements=[LifeCell.lattice('cells', space, alive=values, born=tuple(born), survive=tuple(survive))],
    )


@register_model(
    field='cellular automata',
    tags=('lattice', 'opinion', 'voting'),
    references=('Charbonneau 2017',),
    defaults={'until': 50, 'dt': 1},
)
def majority_rule(
    shape: Sequence[int] = (100, 100),
    *,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voters that take the majority opinion of their neighbourhood, which coarsens into domains.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a voter starts with opinion 1.

        random_state:
            The seed or generator of the initial opinions.

    Returns:
        system:
            The model, with the opinions `voters.opinion`.

    Examples:
        >>> from skcomplex.models.lattices import majority_rule
        >>> majority_rule((10, 10)).elements[0].size
        100
    """
    values = _random_cells(shape, density, random_state)
    return System(elements=[Voter.lattice('voters', LatticeSpace(tuple(shape)), opinion=values)])


@register_model(
    field='cellular automata',
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
        >>> from skcomplex.models.lattices import langtons_ant
        >>> int((langtons_ant((9, 9)).elements[0].values['ant'] >= 0).sum())
        1
    """
    ant = np.full(tuple(shape), -1.0)
    ant[shape[0] // 2, shape[1] // 2] = heading
    return System(elements=[AntCell.lattice('cells', LatticeSpace(tuple(shape)), ant=ant)])


@register_model(
    field='transport',
    tags=('lattice', 'traffic', 'jams', 'stochastic'),
    references=('Nagel and Schreckenberg 1992', 'Charbonneau 2017'),
    defaults={'until': 200, 'dt': 1},
)
def nagel_schreckenberg(
    length: int = 1000,
    *,
    density: float = 0.2,
    slowdown: float = 0.3,
    vmax: int = 5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Nagel-Schreckenberg traffic model on a ring road, where jams emerge above a critical density.

    Args:
        length:
            The number of cells of the road.

        density:
            The fraction of cells with a car.

        slowdown:
            The probability of random braking.

        vmax:
            The largest velocity.

        random_state:
            The seed or generator of the initial positions of the cars.

    Returns:
        system:
            The model, with the velocities `road.velocity`, -1 on empty cells.

    Examples:
        >>> from skcomplex.models.lattices import nagel_schreckenberg
        >>> int((nagel_schreckenberg(100, density=0.1).elements[0].values['velocity'] >= 0).sum())
        10
    """
    rng = np.random.default_rng(random_state)
    velocity = np.full(length, -1.0)
    velocity[rng.choice(length, size=round(density * length), replace=False)] = 0.0
    road = Road.lattice('road', LatticeSpace(length), velocity=velocity, slowdown=slowdown, vmax=vmax)
    return System(elements=[road])


@register_model(
    field='cellular automata',
    tags=('lattice', 'excitable media', 'spirals'),
    references=('Gerhardt and Schuster 1989', 'Charbonneau 2017'),
    defaults={'until': 200, 'dt': 1},
)
def hodgepodge(
    shape: Sequence[int] = (100, 100),
    *,
    n: float = 100.0,
    k1: float = 2.0,
    k2: float = 3.0,
    g: float = 28.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the hodgepodge machine, a model of a chemical reaction whose waves form spirals.

    Args:
        shape:
            The number of rows and columns.

        n:
            The level of illness.

        k1:
            The weight of infected neighbours for healthy cells.

        k2:
            The weight of ill neighbours for healthy cells.

        g:
            The speed of the infection.

        random_state:
            The seed or generator of the initial levels.

    Returns:
        system:
            The model, with the levels `cells.level`.

    Examples:
        >>> from skcomplex.models.lattices import hodgepodge
        >>> hodgepodge((10, 10)).elements[0].values['g']
        28.0
    """
    levels = np.random.default_rng(random_state).integers(0, int(n) + 1, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape))
    return System(elements=[HodgepodgeCell.lattice('cells', space, level=levels, n=n, k1=k1, k2=k2, g=g)])
