"""Define the law that sets cells from the ordered values of their neighbourhoods."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ...errors import ValidationError
from ...types import IntArray
from ..spaces import BaseNeighborhood, BaseSpace, LatticeNeighborhood, LatticeSpace, NetworkSpace
from ..spaces._discrete_time import _DiscreteTime
from ._base import BaseLaw
from ._linear_rule_spec import LinearRuleSpec
from ._outer_totalistic_spec import OuterTotalisticSpec
from ._requirement_spec import RequirementSpec
from ._table_spec import TableSpec
from ._totalistic_spec import TotalisticSpec

SCHEDULES = ('synchronous', 'random-sequential', 'random-site', 'checkerboard', 'colored')
TABLE_AXES = 2


@dataclass(frozen=True)
class _LocalRuleLaw(BaseLaw):
    """A law that sets the next value of cells from their ordered neighbourhoods.

    A function rule receives one array per neighbour of every read state, in neighbourhood order, with one value per
    site. It also receives the parameters of the element it names, `random`, a uniform number per site, when the rule is
    stochastic, and `degree`, the number of neighbours of every site, when it names it. It returns the next values of
    the target, or one array per target. A table or a totalistic rule may stand in place of the function.

    A run computes one synchronous step. Its input holds `space`, the space of the sites, the values of every read state
    by the name of its symbol, and the arguments a function rule names. Sites outside a lattice with fixed edges read
    the fill value of the space.

    Unlike `UpdateLaw`, which gives the next value by an expression of one member, it reads the ordered neighbourhood of
    each cell. Choose it when the next value of a cell depends on the exact pattern around it.

    It appears in many fields. Conway's Game of Life, elementary cellular automata, models of excitable tissue, and
    models of traffic on a ring road follow local rules.

    Args:
        targets:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        over:
            The neighbourhood whose values the rule reads: a lattice neighbourhood of offsets, an adjacency on a
            network, or a neighbourhood of a space outside the core, such as a memory of group elements on a group
            space, which may move with the group element that the first read state holds at every site.

        rule:
            A function, a `TableSpec`, a `TotalisticSpec` rule, an `OuterTotalisticSpec` rule, a `LinearRuleSpec`, or an
            object whose `get_table()` returns a `TableSpec`, such as a rule code.

        reads:
            The lattice states whose neighbourhoods a function rule reads, or `None` for the targets.

        schedule:
            `'synchronous'` to update every site from the previous configuration, `'random-sequential'` to update the
            sites one at a time in a random order, `'random-site'` to update as many randomly chosen sites,
            `'checkerboard'` to update the two sublattices of a square lattice in turn, or `'colored'` to update the
            classes of a proper coloring of the neighbour graph in turn.

        stochastic:
            Whether a function rule receives the uniform random array `random`.

        label:
            The name of the law, or the empty string for a law without one.

    Raises:
        ValidationError:
            The schedule or the rule spec is invalid, or a run lacks the space or a read state.
    """

    law_tag: ClassVar[str] = 'update'

    targets: Expression | tuple[Expression, ...]
    over: BaseNeighborhood
    rule: object
    _: KW_ONLY
    reads: tuple[Expression, ...] | None = None
    schedule: str = 'synchronous'
    stochastic: bool = False
    label: str = ''

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'UpdateLaw'

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice states the rule updates."""
        return self.targets if isinstance(self.targets, tuple) else (self.targets,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the rule requires of a system.

        It requires discrete time on a space of sites, such as a lattice, a network, or a Cayley graph.
        """
        return RequirementSpec(time=(_DiscreteTime,), space=(LatticeSpace, NetworkSpace), sites=True)

    def _check(self) -> None:
        """Raise an error for an unknown schedule or an invalid rule spec."""
        super()._check()
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValidationError(error_msg)
        if isinstance(self.rule, BaseSpec):
            self.rule._check()

    def _get_rule(self) -> object:
        """Return a rule code as its table, and any other rule unchanged.

        The table comes from its `get_table()` method.
        """
        convert = getattr(self.rule, 'get_table', None)
        return convert() if not callable(self.rule) and callable(convert) else self.rule

    def _get_neighborhood(self, space: BaseSpace) -> BaseNeighborhood:
        """Return the neighbourhood the rule reads on a prepared space.

        A neighbourhood with lattice offsets becomes the list of its offsets on the space. For an outer totalistic
        rule, the site itself comes first, where the neighbourhood puts it or else as the zero offset.
        """
        outer = isinstance(self._get_rule(), OuterTotalisticSpec)
        centered = self.over._get_centered() if outer else None
        if centered is not None:
            return centered
        offsets = self.over._get_offsets(space.n_dim_)
        if offsets is None and outer:
            error_msg = f'A run reads an outer totalistic rule on offsets or a memory set, not on {self.over!r}.'
            raise ValidationError(error_msg)
        if offsets is None:
            return self.over
        return LatticeNeighborhood(((0,) * len(offsets[0]), *offsets) if outer else offsets)

    def _get_table(self, space: BaseSpace, values: npt.NDArray[np.generic]) -> IntArray:
        """Return the neighbour table.

        For a translated memory, it is the table of every configuration at its values.

        Args:
            space:
                The space of the sites.

            values:
                The values of the first read state, one per site on the last axis.

        Returns:
            table:
                One row per site and one column per neighbour, with leading axes of the configurations when translated.
        """
        neighborhood = self._get_neighborhood(space)
        if not neighborhood._is_translated():
            return space._get_index_table(neighborhood)
        translations = space._get_translation_table(neighborhood)
        size = translations.shape[0]
        elements = np.rint(np.asarray(values, dtype=float))
        named = (elements >= 0) & (elements < size)
        table = translations[np.arange(size), np.where(named, elements, 0).astype(np.intp)]
        return np.where(named[..., None], table, size)

    def _get_linear_values(self, *columns: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Return the linear combination of one column of values per neighbour.

        It is reduced modulo the field size.
        """
        rule = self._get_rule()
        if not isinstance(rule, LinearRuleSpec) or len(columns) != len(rule.coefficients):
            error_msg = f'The rule {rule!r} is not a linear rule of {len(columns)} neighbours.'
            raise ValidationError(error_msg)
        image = np.stack([np.asarray(column) for column in columns], axis=-1) @ np.asarray(rule.coefficients)
        characteristic = rule.characteristic
        return np.rint(image).astype(np.int64) % characteristic if characteristic else image

    @staticmethod
    def _get_totalistic_values(rule: TotalisticSpec, patterns: IntArray) -> npt.NDArray[np.generic]:
        """Return the next value of every row of neighbourhood values from its sum."""
        sums = patterns.sum(axis=1)
        if callable(rule.rule):
            return np.broadcast_to(np.asarray(rule.rule(sums)), sums.shape)
        return np.asarray([rule.rule.get(total, rule.default) for total in sums.tolist()])

    @staticmethod
    def _get_outer_values(rule: OuterTotalisticSpec, patterns: IntArray) -> npt.NDArray[np.generic]:
        """Return the next value of every row of neighbourhood values.

        It comes from its first cell and the sum of the others.
        """
        centers, sums = patterns[:, 0], patterns[:, 1:].sum(axis=1)
        if callable(rule.rule):
            return np.broadcast_to(np.asarray(rule.rule(centers, sums)), sums.shape)
        pairs = zip(centers.tolist(), sums.tolist(), strict=True)
        return np.asarray([rule.rule.get(pair, rule.default) for pair in pairs])

    def _get_next_values(self, patterns: IntArray, **keywords: object) -> object:
        """Return the next value of every row of neighbourhood values.

        The cells are in the order the rule reads them.
        """
        rule = self._get_rule()
        if isinstance(rule, LinearRuleSpec):
            return self._get_linear_values(*patterns.T)
        if isinstance(rule, TableSpec):
            return np.asarray([rule.mapping.get(tuple(row), rule.default) for row in patterns.tolist()])
        if isinstance(rule, TotalisticSpec):
            return self._get_totalistic_values(rule, patterns)
        if isinstance(rule, OuterTotalisticSpec):
            return self._get_outer_values(rule, patterns)
        if callable(rule):
            return rule(*patterns.T, **keywords)
        error_msg = (
            f'The rule {rule!r} is not a TableSpec, a totalistic rule, a linear rule, a rule code with a get_table() '
            'method, or a function.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_cells(values: object, shape: tuple[int, ...]) -> npt.NDArray[np.generic]:
        """Return next values in the shape of the cells.

        They are integers when every one of them is an integer.
        """
        size = int(np.prod(shape))
        cells = np.broadcast_to(np.asarray(values, dtype=float), (size,)).reshape(shape)
        return cells.astype(np.int64) if np.all(cells == np.rint(cells)) else cells

    def _get_value(self, inputs: object) -> object:
        """Return the next values of the cells.

        A law of several targets returns a tuple of them.

        Args:
            inputs:
                A mapping with `space`, the values of every read state by the name of its symbol, and the keyword
                arguments a function rule names.
        """
        keywords = dict(inputs) if isinstance(inputs, Mapping) else {}
        space = keywords.pop('space', None)
        names = [str(symbol) for symbol in (self.reads or self._get_targets())]
        missing = [name for name in names if name not in keywords]
        if not isinstance(space, BaseSpace) or missing:
            error_msg = f'A local rule runs on a mapping with `space` and its states {names}, but got {inputs!r}.'
            raise ValidationError(error_msg)
        if not hasattr(space, 'size_'):
            space.prepare()
        states = [np.asarray(keywords.pop(name)) for name in names]
        table = self._get_table(space, states[0])
        columns = []
        for values in states:
            fill = np.full((*values.shape[:-1], 1), getattr(space, 'fill', 0), dtype=values.dtype)
            padded = np.concatenate([values, fill], axis=-1)
            if table.ndim == TABLE_AXES:
                columns.append(padded[..., table])
            else:
                gathered = np.take_along_axis(padded, table.reshape(*table.shape[:-2], -1), axis=-1)
                columns.append(gathered.reshape(table.shape))
        cells = np.concatenate(columns, axis=-1)
        shape = cells.shape[:-1]
        patterns = cells.reshape(-1, cells.shape[-1])
        if callable(self._get_rule()):
            result = self._get_next_values(patterns, **keywords)
        else:
            rows, inverse = np.unique(patterns.astype(np.int64), axis=0, return_inverse=True)
            result = np.asarray(self._get_next_values(rows))[inverse.ravel()]
        return (
            tuple(self._get_cells(item, shape) for item in result)
            if isinstance(result, tuple)
            else self._get_cells(result, shape)
        )
