"""Expand small groups into one equation per member, with every reduction written as an explicit sum."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, max_symbolic, min_symbolic

from ..base import BaseConfig
from ..core.laws.utils import where
from ..engine._utils import components_of, domain_of, symbol
from ..engine.problems import ProblemSpec
from ..engine.problems._reduction_kernel import ReductionKernel
from ..errors import ValidationError
from ..specs import EntrySpec


class _Expansion(BaseConfig[ProblemSpec, tuple[set[str], int]]):
    """Symbols of every group member and the explicit value of every reduction of one problem.

    It prepares on a compiled problem and runs on the names of the groups of one owner and one member, setting
    `expansion_`, the substitution that writes the equations of that member out.

    Args:
        parameter_values:
            The parameter vector whose values the member parameters take.

    Attributes:
        expansion_:
            The symbols of the member and the value of every reduction, by the symbol they replace, set by `run`.
    """

    expansion_: dict[Expression, Expression]

    def _check(self) -> None:
        """Accept every setting, since an expansion has none to validate; an expansion with settings validates them."""

    def __init__(self, parameter_values: npt.NDArray[np.float64] | None = None) -> None:
        self.parameter_values = parameter_values

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Keep the problem, start the symbols and values of its members, and find the symbols of single components.

        Args:
            context:
                The compiled problem whose groups are expanded.

        Raises:
            ValidationError:
                No problem is given.
        """
        if context is None:
            error_msg = 'The expansion of groups prepares on a compiled problem. Pass one.'
            raise ValidationError(error_msg)
        self.problem_ = context
        self.parameters_ = np.asarray(
            self.parameter_values if self.parameter_values is not None else np.zeros(0),
            dtype=np.float64,
        )
        self.symbols_: dict[tuple[str, int, int], Expression] = {}
        self.values_: dict[Expression, float] = {}
        self.shared_: dict[tuple[Expression, int | None], Expression] = {}
        self.by_index_: dict[tuple[bool, int], Expression] = {}
        for is_state, layout in ((True, context.states), (False, context.parameters)):
            for entry in layout.entries:
                if entry.members is None and entry.name in context.symbols:
                    for column, item in enumerate(components_of(context.symbols[entry.name])):
                        self.by_index_[(is_state, entry.start + column)] = item

    def _run(self, inputs: tuple[set[str], int] | None) -> None:
        """Set `expansion_`, the substitution of one member: its symbols for those of the named groups.

        Every reduction is written as an explicit sum read by the member, and the values of new member parameters are
        kept in `values_`.

        Args:
            inputs:
                The names of the groups of the owner, and the member.

        Raises:
            ValidationError:
                No names and member are given.
        """
        if inputs is None:
            error_msg = 'The expansion of groups runs on the names of the groups of an owner and a member. Pass them.'
            raise ValidationError(error_msg)
        entry_names, member = inputs
        mapping = self._member_map(entry_names, member)
        mapping.update(
            {
                reduction.symbol: self._value(reduction, member if reduction.weights is not None else None)
                for reduction in self.problem_.reductions
            },
        )
        self.expansion_ = mapping

    def _member_symbol(self, entry: EntrySpec, column: int, member: int) -> Expression:
        """Return the symbol of one component of one member of a group."""
        key = (entry.name, column, member)
        if key not in self.symbols_:
            generic = components_of(self.problem_.symbols[entry.name])[column]
            self.symbols_[key] = symbol(f'{generic}__{member}', domain_of(generic))
            if self.problem_.parameters.has(entry.name):
                self.values_[self.symbols_[key]] = float(
                    self.parameters_[entry.start + member * entry.columns + column],
                )
        return self.symbols_[key]

    def _member_map(self, entry_names: set[str], member: int) -> dict[Expression, Expression]:
        """Return the substitution of the generic symbols of groups by the symbols of one member."""
        mapping = {}
        for layout in (self.problem_.states, self.problem_.parameters):
            for entry in layout.entries:
                if entry.members is None or entry.name not in self.problem_.symbols:
                    continue
                if entry.name not in entry_names:
                    continue
                for column, item in enumerate(components_of(self.problem_.symbols[entry.name])):
                    mapping[item] = self._member_symbol(entry, column, member)
        return mapping

    def _read_symbol(self, reduction: ReductionKernel, item: Expression, member: int) -> Expression:
        """Return what one symbol of a reduction stands for at one reduced member."""
        read = reduction.reads[item]
        nested = next((other for other in self.problem_.reductions if other.symbol == item), None)
        if nested is not None:
            return self._value(nested, member if nested.weights is not None else None)
        if not read.per_member:
            return item
        if read.entry is None:
            return self.by_index_[(read.is_state, int(np.asarray(read.index)[member]))]
        return self._member_symbol(read.entry, read.column, member)

    def _value(self, reduction: ReductionKernel, reader: int | None) -> Expression:
        """Return a reduction as an explicit expression of the member symbols, for one reader with weights."""
        key = (reduction.symbol, reader)
        if key in self.shared_:
            return self.shared_[key]
        terms, selected = [], []
        for member in range(reduction.size):
            mapping = {item: self._read_symbol(reduction, item, member) for item in reduction.reads}
            terms.append(SR(reduction.argument).subs(mapping))
            condition = SR(reduction.condition).subs(mapping) if reduction.condition is not None else None
            selected.append(condition)
        self.shared_[key] = self._reduce(reduction, terms, selected, reader)
        return self.shared_[key]

    @staticmethod
    def _chosen(term: Expression, condition: Expression | None, otherwise: object) -> Expression:
        """Return a member's term where its condition holds and another value elsewhere."""
        return term if condition is None else where(condition, term, otherwise)

    @staticmethod
    def _reduce(
        reduction: ReductionKernel,
        terms: list[Expression],
        selected: list[Expression | None],
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
