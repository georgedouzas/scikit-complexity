"""Automata models: life-like rules, noisy rules, discrete diffusion, and an unbounded alphabet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, ParameterError, parameter, state
from ....core.model import Element, LatticeSpace, Law, LocalRule, Neighborhood, Symbols, System
from ....core.runtime import register_model
from ._automata import LifeCell

type Array = npt.NDArray[np.float64]

LIFE_RULE = re.compile(r'^B(?P<born>[0-8]*)/S(?P<survive>[0-8]*)$')
PREDECESSORS = Neighborhood(((-1,), (0,)))


def _parse_life_rule(rule: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the birth and survival counts of a life-like rule in the notation `B36/S23`."""
    match = LIFE_RULE.match(rule.upper())
    if match is None:
        error_msg = f'A life-like rule reads like `B36/S23`, with digits 0 to 8, but it is {rule!r}.'
        raise ParameterError(error_msg)
    return tuple(int(digit) for digit in match['born']), tuple(int(digit) for digit in match['survive'])


class NoisyElementaryCell(Element):
    """Cell of a one-dimensional automaton that follows an elementary rule, with each output flipped with a probability.

    Args:
        label:
            The label of the cell or the lattice group.

        alive:
            The initial value, 0 or 1.

        noise:
            The probability that the output of the rule is flipped.

        rule:
            The Wolfram number of the rule.

    Examples:
        >>> from skcomplex.models.computer_science.automata import NoisyElementaryCell
        >>> NoisyElementaryCell('cell', rule=184).rule
        184
    """

    alive: float = state(default=0, domain=Domain.BOOLEAN)
    noise: float = parameter(0.01, domain=Domain.NONNEGATIVE)
    rule: int = 18

    def laws(self, s: Symbols) -> list[Law]:
        """Return the elementary rule followed by a random flip.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        number = self.rule

        def noisy(left: Array, center: Array, right: Array, *, noise: Any, random: Array) -> Array:  # noqa: ANN401
            output = (number >> (4 * left + 2 * center + right).astype(int)) & 1
            return np.where(random < noise, 1 - output, output).astype(float)

        return [LocalRule(s.alive, Neighborhood.radius(), noisy, stochastic=True)]


class DiffusionCell(Element):
    """Cell of a lattice that exchanges a share of its content with each of its von Neumann neighbours.

    Args:
        label:
            The label of the cell or the lattice group.

        level:
            The initial content.

        r:
            The share of the difference with a neighbour exchanged in a step, below `1 / (2 d)` in `d` dimensions.

    Examples:
        >>> from skcomplex.models.computer_science.automata import DiffusionCell
        >>> DiffusionCell('cell', r=0.2).r
        0.2
    """

    level: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r: float = parameter(0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the discrete diffusion rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """

        def diffuse(level: Array, *around: Array, r: Any) -> Array:  # noqa: ANN401
            return level + r * (sum(around) - len(around) * level)

        return [LocalRule(s.level, Neighborhood.von_neumann(center=True), diffuse)]


class PascalCell(Element):
    """Cell over the nonnegative integers whose next value is the sum of its own and its left neighbour's values.

    Args:
        label:
            The label of the cell or the lattice group.

        value:
            The initial value, a nonnegative integer.

    Examples:
        >>> from skcomplex.models.computer_science.automata import PascalCell
        >>> PascalCell('cell', value=3).value
        3
    """

    value: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the additive rule over the integers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.value, PREDECESSORS, lambda left, center: left + center)]


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'life', 'replicator'),
    references=('Bell 1994', 'Charbonneau 2017', 'Downey 2018'),
    defaults={'until': 100, 'dt': 1},
)
def highlife(
    shape: Sequence[int] = (100, 100),
    *,
    rule: str = 'B36/S23',
    density: float = 0.3,
    initial: npt.ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return HighLife, `B36/S23`, or any other life-like automaton of the planar zoo by its rule, on a torus.

    HighLife adds birth on six neighbours to Conway's rule, which gives it a small self-replicating pattern. Other
    members of the zoo are Seeds, `B2/S`, Day and Night, `B3678/S34678`, and Life without death, `B3/S012345678`.

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

    Returns:
        system:
            The model, with the cells `cells.alive`.

    Raises:
        ParameterError:
            The rule is not in the notation `B<births>/S<survivals>`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import highlife
        >>> highlife((10, 10), rule='B2/S').elements[0].values['born']
        (2,)
    """
    born, survive = _parse_life_rule(rule)
    values = _random_cells(shape, density, random_state) if initial is None else np.asarray(initial, dtype=float)
    space = LatticeSpace(tuple(shape))
    return System(components=[LifeCell.lattice('cells', space, alive=values, born=born, survive=survive)])


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'elementary', 'noise', 'stochastic'),
    references=('Lindgren 2024',),
    defaults={'until': 100, 'dt': 1},
)
def noisy_automata(
    rule: int = 18,
    *,
    noise: float = 0.01,
    size: int = 101,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return an elementary cellular automaton whose outputs are flipped at random with a small probability.

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

    Returns:
        system:
            The model, with the cells `cells.alive`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import noisy_automata
        >>> noisy_automata(184, size=11).elements[0].size
        11
    """
    values = _random_cells((size,), density, random_state)
    space = LatticeSpace(size)
    return System(components=[NoisyElementaryCell.lattice('cells', space, alive=values, noise=noise, rule=rule)])


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'diffusion', 'conservation'),
    references=('Downey 2018',),
    defaults={'until': 100, 'dt': 1},
)
def diffusion_automaton(shape: Sequence[int] = (50, 50), *, r: float = 0.1, width: int = 10) -> System:
    """Return a cellular automaton of diffusion, `c' = c + r Σ (c_neighbour - c)`, started from a square of ones.

    The total content is conserved on the torus, and the square spreads into a Gaussian bump.

    Args:
        shape:
            The number of rows and columns of the torus.

        r:
            The share of the difference with a neighbour exchanged in a step.

        width:
            The side of the central square of cells that start with content 1.

    Returns:
        system:
            The model, with the contents `cells.level`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import diffusion_automaton
        >>> float(diffusion_automaton((20, 20), width=4).elements[0].values['level'].sum())
        16.0
    """
    level = np.zeros(tuple(shape))
    rows, columns = (shape[0] - width) // 2, (shape[1] - width) // 2
    level[rows : rows + width, columns : columns + width] = 1.0
    return System(components=[DiffusionCell.lattice('cells', LatticeSpace(tuple(shape)), level=level, r=r)])


@register_model(
    field='computer_science/automata',
    tags=('lattice', 'integers', 'additive', 'pascal triangle'),
    references=('Ceccherini-Silberstein and Coornaert 2023',),
    defaults={'until': 50, 'dt': 1},
)
def unbounded_alphabet_automaton(size: int = 101) -> System:
    """Return an additive automaton over the nonnegative integers, `x'(n) = x(n - 1) + x(n)`, from a single 1.

    Its alphabet has no bound: from a single 1 the cells after `t` steps hold the binomial coefficients `C(t, k)`, so
    every row of Pascal's triangle appears.

    Args:
        size:
            The number of cells of the ring.

    Returns:
        system:
            The model, with the cells `cells.value`.

    Examples:
        >>> from skcomplex.models.computer_science.automata import unbounded_alphabet_automaton
        >>> int(unbounded_alphabet_automaton(9).elements[0].values['value'].sum())
        1
    """
    value = np.zeros(size)
    value[size // 2] = 1.0
    return System(components=[PascalCell.lattice('cells', LatticeSpace(size), value=value)])
