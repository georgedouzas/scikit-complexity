"""Define local rule laws.

They give the next values of lattice cells from the ordered values of their neighbourhoods.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import BaseSpec, LinearRuleSpec, OuterTotalisticSpec, RequirementSpec, TableSpec, TotalisticSpec
from ...types import IntArray
from ..spaces import (
    BaseNeighborhood,
    BaseSpace,
    CayleySpace,
    DiscreteTime,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
    NetworkSpace,
    StateTranslationNeighborhood,
)
from ._base import BaseLaw

SCHEDULES = ('synchronous', 'random-sequential', 'random-site', 'checkerboard', 'colored')
IDENTITY = '1'
TABLE_AXES = 2


class LocalRuleLaw(BaseLaw):
    """Next value of lattice cells from the ordered values of their neighbourhoods.

    A function rule receives one array per neighbour of every read state, in neighbourhood order, with one value per
    site, and keyword arguments for the parameters of the element that it names, plus `random`, an array of uniform
    numbers per site, when the rule is stochastic, and `degree`, the number of neighbours of every site besides the
    site itself, when it names it. It returns the next values of the target, or a tuple with one array per target.

    A run computes one synchronous step. Its input is a mapping with `space`, the space of the sites, the values of
    every read state under the name of its symbol, with one value per site or one row of values per configuration, and
    the keyword arguments a function rule names. Sites outside a lattice with fixed edges or outside a ball read the
    fill value of the space.

    Args:
        targets:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        over:
            The neighbourhood whose values the rule reads: a lattice neighbourhood of offsets, an adjacency on a
            network, a memory of group elements on a group space, or a memory translated by the group element that
            the first read state holds at every site, `StateTranslationNeighborhood`.

        rule:
            A function, a `TableSpec`, a `TotalisticSpec` rule, an `OuterTotalisticSpec` rule, a `LinearRuleSpec`, or an
            object whose `table()` returns a `TableSpec`, such as a rule code.

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

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import LocalRuleLaw
        >>> from skcomplex.core.specs import LinearRuleSpec, TotalisticSpec
        >>> from skcomplex.core.components.spaces import BallNeighborhood, LatticeNeighborhood, LatticeSpace
        >>> cell = var('cell')
        >>> box = BallNeighborhood(n_dim=1, center='inline')
        >>> LocalRuleLaw(cell, box, TotalisticSpec({2: 1})).prepare().targets_
        (cell,)
        >>> parity = LocalRuleLaw(cell, box, TotalisticSpec(lambda total: total % 2)).prepare()
        >>> parity.run({'space': LatticeSpace(5), 'cell': [0, 0, 1, 0, 0]}).value_.tolist()
        [0, 1, 1, 1, 0]
        >>> around = LatticeNeighborhood(((-1,), (1,)))
        >>> rule90 = LocalRuleLaw(cell, around, LinearRuleSpec((1, 1), 'GF(2)')).prepare()
        >>> rule90.run({'space': LatticeSpace(6), 'cell': [0, 0, 1, 0, 0, 0]}).value_.tolist()
        [0, 1, 0, 1, 0, 0]
    """

    law_tag: ClassVar[str] = 'update'

    def __init__(
        self,
        targets: Expression | tuple[Expression, ...],
        over: BaseNeighborhood,
        rule: object,
        *,
        reads: tuple[Expression, ...] | None = None,
        schedule: str = 'synchronous',
        stochastic: bool = False,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.targets = targets
        self.over = over
        self.rule = rule
        self.reads = reads
        self.schedule = schedule
        self.stochastic = stochastic

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice states the rule updates."""
        return self.targets if isinstance(self.targets, tuple) else (self.targets,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the rule requires of a system.

        It requires discrete time on a lattice, a network, or a Cayley graph.
        """
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace, CayleySpace))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for an unknown schedule or an invalid rule spec.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValidationError(error_msg)
        if isinstance(self.rule, BaseSpec):
            self.rule._check()

    def _get_rule(self) -> object:
        """Return a rule code as its table, and any other rule unchanged.

        The table comes from its `table()` method.
        """
        convert = getattr(self.rule, 'table', None)
        return convert() if not callable(self.rule) and callable(convert) else self.rule

    def _get_neighborhood(self) -> BaseNeighborhood:
        """Return the neighbourhood the rule reads.

        For an outer totalistic rule, the site itself comes first.
        """
        if not isinstance(self._get_rule(), OuterTotalisticSpec):
            return self.over
        if isinstance(self.over, MemoryNeighborhood):
            return MemoryNeighborhood((IDENTITY, *self.over.elements))
        if not isinstance(self.over, LatticeNeighborhood):
            error_msg = f'A run reads an outer totalistic rule on offsets or a memory set, not on {self.over!r}.'
            raise ValidationError(error_msg)
        offsets = self.over.prepare().offsets_
        return LatticeNeighborhood(((0,) * len(offsets[0]), *offsets))

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
        neighborhood = self._get_neighborhood()
        if not isinstance(neighborhood, StateTranslationNeighborhood):
            return space._index_table(neighborhood)
        translations = space._translation_table(neighborhood)
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
            f'The rule {rule!r} is not a TableSpec, a totalistic rule, a linear rule, a rule code with a table() '
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

    def _run(self, inputs: object) -> None:
        """Set `value_` to the next values of the cells.

        A law of several targets sets a tuple of them.

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
        self.value_ = (
            tuple(self._get_cells(item, shape) for item in result)
            if isinstance(result, tuple)
            else self._get_cells(result, shape)
        )
