"""Expand small groups into one equation per member, with every reduction written as an explicit sum."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, max_symbolic, min_symbolic

from ..core.laws.utils import where
from ..engine._utils import components_of, domain_of, symbol
from ..specs import EntrySpec, ProblemSpec


@dataclass
class _Expansion:
    """Symbols of every group member and the explicit value of every reduction, built for one problem."""

    problem: ProblemSpec
    parameter_values: npt.NDArray[np.float64]
    symbols: dict[tuple[str, int, int], Expression] = field(default_factory=dict)
    values: dict[Expression, float] = field(default_factory=dict)
    shared: dict[tuple[Expression, int | None], Expression] = field(default_factory=dict)

    @functools.cached_property
    def by_index(self) -> dict[tuple[bool, int], Expression]:
        """Symbols of single components by their flat index, so element sets find their elements."""
        found = {}
        for is_state, layout in ((True, self.problem.states), (False, self.problem.parameters)):
            for entry in layout.entries:
                if entry.members is None and entry.name in self.problem.symbols:
                    for column, item in enumerate(components_of(self.problem.symbols[entry.name])):
                        found[(is_state, entry.start + column)] = item
        return found

    def member_symbol(self, entry: EntrySpec, column: int, member: int) -> Expression:
        """Return the symbol of one component of one member of a group."""
        key = (entry.name, column, member)
        if key not in self.symbols:
            generic = components_of(self.problem.symbols[entry.name])[column]
            self.symbols[key] = symbol(f'{generic}__{member}', domain_of(generic))
            if self.problem.parameters.has(entry.name):
                self.values[self.symbols[key]] = float(
                    self.parameter_values[entry.start + member * entry.columns + column],
                )
        return self.symbols[key]

    def member_map(self, entry_names: set[str], member: int) -> dict[Expression, Expression]:
        """Return the substitution of the generic symbols of groups by the symbols of one member."""
        mapping = {}
        for layout in (self.problem.states, self.problem.parameters):
            for entry in layout.entries:
                if entry.members is None or entry.name not in self.problem.symbols:
                    continue
                if entry.name not in entry_names:
                    continue
                for column, item in enumerate(components_of(self.problem.symbols[entry.name])):
                    mapping[item] = self.member_symbol(entry, column, member)
        return mapping

    def _read_symbol(self, reduction: Any, item: Expression, member: int) -> Expression:  # noqa: ANN401
        """Return what one symbol of a reduction stands for at one reduced member."""
        read = reduction.reads[item]
        nested = next((other for other in self.problem.reductions if other.symbol == item), None)
        if nested is not None:
            return self.value(nested, member if nested.weights is not None else None)
        if not read.per_member:
            return item
        if read.entry is None:
            return self.by_index[(read.is_state, int(np.asarray(read.index)[member]))]
        return self.member_symbol(read.entry, read.column, member)

    def value(self, reduction: Any, reader: int | None) -> Expression:  # noqa: ANN401
        """Return a reduction as an explicit expression of the member symbols, for one reader with weights."""
        key = (reduction.symbol, reader)
        if key in self.shared:
            return self.shared[key]
        terms, selected = [], []
        for member in range(reduction.size):
            mapping = {item: self._read_symbol(reduction, item, member) for item in reduction.reads}
            terms.append(SR(reduction.argument).subs(mapping))
            condition = SR(reduction.condition).subs(mapping) if reduction.condition is not None else None
            selected.append(condition)
        self.shared[key] = self._reduce(reduction, terms, selected, reader)
        return self.shared[key]

    @staticmethod
    def _chosen(term: Expression, condition: Expression | None, otherwise: Any) -> Expression:  # noqa: ANN401
        """Return a member's term where its condition holds and another value elsewhere."""
        return term if condition is None else where(condition, term, otherwise)

    @staticmethod
    def _reduce(
        reduction: Any,  # noqa: ANN401
        terms: list[Expression],
        selected: list[Any],
        reader: int | None,
    ) -> Expression:
        """Return the explicit value of a reduction from the terms and conditions of its members."""
        counted = sum((_Expansion._chosen(SR(1), condition, 0) for condition in selected), SR(0))
        if reduction.kind == 'count':
            return counted
        if reduction.weights is not None:
            row = reduction.weights.toarray()[reader if reader is not None else 0]
            weighted = sum(
                (
                    SR(float(weight)) * _Expansion._chosen(term, condition, 0)
                    for weight, term, condition in zip(row, terms, selected, strict=True)
                    if weight
                ),
                SR(0),
            )
            return weighted / counted if reduction.kind == 'mean' else weighted
        summed = sum(
            (_Expansion._chosen(term, condition, 0) for term, condition in zip(terms, selected, strict=True)),
            SR(0),
        )
        if reduction.kind == 'sum':
            return summed
        if reduction.kind == 'mean':
            return summed / counted
        if reduction.kind == 'var':
            squares = sum(
                (_Expansion._chosen(term**2, condition, 0) for term, condition in zip(terms, selected, strict=True)),
                SR(0),
            )
            return squares / counted - (summed / counted) ** 2
        extreme = max_symbolic if reduction.kind == 'max' else min_symbolic
        bound = SR(-1e300) if reduction.kind == 'max' else SR(1e300)
        return extreme(
            *(_Expansion._chosen(term, condition, bound) for term, condition in zip(terms, selected, strict=True)),
        )
