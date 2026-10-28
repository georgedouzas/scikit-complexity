"""Cellular automaton on a finite universe as a map on configurations, with the helpers of finite automata."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ....core.laws.utils import OuterTotalistic, Table, Totalistic
from ....core.spaces import BaseSpace, LatticeNeighborhood, MemoryNeighborhood
from ....errors import ValidationError
from ....types import Indices, IntArray

MAX_PATTERNS = 2**20
IDENTITY = '1'

type Reach = LatticeNeighborhood | MemoryNeighborhood


def _fitted(space: BaseSpace) -> BaseSpace:
    """Return a space of sites, preparing it on its first use."""
    if not hasattr(space, 'size_'):
        space.check().prepare()
    return space


def _table(space: BaseSpace, reach: Reach) -> Indices:
    """Return the site of every neighbour of every site of a space, with the number of sites outside the space."""
    return _fitted(space)._index_table(reach)


def _count(reach: Reach) -> int:
    """Return the number of cells a neighbourhood of offsets or a memory set reads."""
    return len(reach.check().prepare().offsets_ if isinstance(reach, LatticeNeighborhood) else reach.elements)


def _size(space: BaseSpace) -> int:
    """Return the number of sites of a space."""
    return int(_fitted(space).size_ or 0)


def _codes(values: IntArray, states: int) -> IntArray:
    """Return the code of every row of values, read as digits in the base of the number of states."""
    weights = states ** np.arange(values.shape[-1] - 1, -1, -1, dtype=np.int64)
    return values @ weights


def _patterns(states: int, width: int) -> IntArray:
    """Return every pattern of values of a number of cells, in the order of their codes."""
    if states**width > MAX_PATTERNS:
        error_msg = f'A rule on {width} cells with {states} values has more than {MAX_PATTERNS} patterns.'
        raise ValidationError(error_msg)
    return np.asarray(list(itertools.product(range(states), repeat=width)), dtype=np.int64).reshape(-1, width)


@dataclass(frozen=True)
class Automaton:
    """Cellular automaton on a finite universe: a periodic lattice, a finite quotient of a group, or a ball.

    The automaton maps a configuration, one integer value from `0` to `states - 1` per site, to the configuration whose
    value at every site is the rule applied to the values of the neighbourhood of the site, as `LocalRuleLaw`s do.
    Sites outside a lattice with fixed edges or outside a ball read the fill value of the space.

    Args:
        space:
            The lattice or the group space.

        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space.

        rule:
            A `Table`, a `Totalistic` rule, an `OuterTotalistic` rule, an object whose `table()` returns a `Table`,
            or a function of one array per neighbour that returns the next values.

        states:
            The number of values of a cell.

    Examples:
        >>> from skcomplex.core.laws.utils import Totalistic
        >>> from skcomplex.core.spaces import BoxNeighborhood, LatticeSpace
        >>> from skcomplex.models.computer_science.automata import Automaton
        >>> parity = Automaton(LatticeSpace(5), BoxNeighborhood(), Totalistic(lambda total: total % 2))
        >>> parity.step([0, 0, 1, 0, 0]).tolist()
        [0, 1, 1, 1, 0]
    """

    space: BaseSpace
    neighborhood: Reach
    rule: object
    states: int = 2

    def _rule_table(self) -> object:
        """Return a rule code as its table, through its `table()` method, and any other rule unchanged."""
        convert = getattr(self.rule, 'table', None)
        return convert() if not callable(self.rule) and callable(convert) else self.rule

    @staticmethod
    def _with_center(neighborhood: Reach) -> Reach:
        """Return a neighbourhood with the site itself first, as outer totalistic rules read it."""
        if isinstance(neighborhood, MemoryNeighborhood):
            return MemoryNeighborhood((IDENTITY, *neighborhood.elements))
        offsets = neighborhood.check().prepare().offsets_
        zero = (0,) * len(offsets[0])
        return LatticeNeighborhood((zero, *offsets))

    def _evaluate(self, patterns: IntArray) -> npt.NDArray[np.generic]:
        """Return the next value of every pattern, one row per pattern in the order the rule reads its cells."""
        rule = self._rule_table()
        if isinstance(rule, Table):
            return np.asarray([rule.mapping.get(tuple(int(item) for item in row), rule.default) for row in patterns])
        if isinstance(rule, Totalistic):
            sums = patterns.sum(axis=1)
            if callable(rule.rule):
                return np.broadcast_to(np.asarray(rule.rule(sums)), sums.shape)
            return np.asarray([rule.rule.get(int(total), rule.default) for total in sums])
        if isinstance(rule, OuterTotalistic):
            centers, sums = patterns[:, 0], patterns[:, 1:].sum(axis=1)
            if callable(rule.rule):
                return np.broadcast_to(np.asarray(rule.rule(centers, sums)), sums.shape)
            pairs = zip(centers.tolist(), sums.tolist(), strict=True)
            return np.asarray([rule.rule.get((center, total), rule.default) for center, total in pairs])
        if callable(rule):
            return np.broadcast_to(np.asarray(rule(*patterns.T)), (patterns.shape[0],))
        error_msg = (
            f'The rule {rule!r} is not a Table, a totalistic rule, a rule code with a table() method, or a function.'
        )
        raise ValidationError(error_msg)

    @property
    def reads(self) -> Reach:
        """Neighbourhood the rule reads, with the site itself first for an outer totalistic rule."""
        outer = isinstance(self._rule_table(), OuterTotalistic)
        return self._with_center(self.neighborhood) if outer else self.neighborhood

    @property
    def size(self) -> int:
        """Number of sites."""
        return _size(self.space)

    def table(self) -> Indices:
        """Return the site of every cell the rule reads at every site, with the number of sites outside the space.

        Returns:
            table:
                One row per site and one column per cell of `reads`.
        """
        return _table(self.space, self.reads)

    def local_map(self) -> IntArray:
        """Return the next value of every pattern of the cells the rule reads, indexed by the code of the pattern.

        Returns:
            values:
                One integer value per pattern.

        Raises:
            ValidationError:
                The rule returns a value outside `0` to `states - 1`.
        """
        patterns = _patterns(self.states, _count(self.reads))
        values = np.asarray(self._evaluate(patterns), dtype=float)
        if values.size and (values.min() < 0 or values.max() >= self.states or not np.all(values == np.rint(values))):
            error_msg = f'The rule returns values outside the integers 0 to {self.states - 1}.'
            raise ValidationError(error_msg)
        return values.astype(np.int64)

    def step(self, configuration: npt.ArrayLike) -> IntArray:
        """Return the image of a configuration, or of one configuration per row.

        Args:
            configuration:
                The values of the sites, or one row of values per configuration.

        Returns:
            image:
                The next values, of the same shape.
        """
        values = np.asarray(configuration, dtype=np.int64)
        fill = np.full((*values.shape[:-1], 1), int(getattr(self.space, 'fill', 0)), dtype=np.int64)
        padded = np.concatenate([values, fill], axis=-1)
        return self.local_map()[_codes(padded[..., self.table()], self.states)]

    def run(self, configuration: npt.ArrayLike, steps: int) -> IntArray:
        """Return the configurations of a number of steps, the initial one first.

        Args:
            configuration:
                The initial values of the sites.

            steps:
                The number of steps.

        Returns:
            configurations:
                One row per step.
        """
        rows = [np.asarray(configuration, dtype=np.int64)]
        for _ in range(steps):
            rows.append(self.step(rows[-1]))
        return np.stack(rows)
