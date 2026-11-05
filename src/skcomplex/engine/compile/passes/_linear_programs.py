"""Merge the linear program laws over shared variables, lower them, and make the functions of the problem solve them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from scipy import sparse

from ....core.laws import LinearProgramLaw
from ....specs import IssueSpec, LayoutSpec, LinearObjectiveSpec, RunStateSpec
from ....specs._base_spec import BaseSpec
from ....types import Array, Indices
from ..._utils import symbol, symbols_of
from ...problems import (
    ExpressionPrinter,
    KernelSpec,
    LinearProgramDualSpec,
    LinearProgramEntrySpec,
    LinearProgramKernel,
    LinearProgramSideSpec,
    LinearProgramSpec,
    ProblemSpec,
    ReferenceSpec,
)
from ...types import LinearProgramBound
from .._base import BasePass
from .._compilation import Compilation, Laws
from ._census import CensusPass

type _Key = tuple[str, int]

ABSOLUTE = 'abs'


@dataclass(frozen=True)
class _BoundLawSpec(BaseSpec):
    """Linear program law with the owner it belongs to and the kernel spec of the component that returns it."""

    spec: KernelSpec
    law: LinearProgramLaw
    owner: str


@dataclass(frozen=True)
class _LinearPartsSpec(BaseSpec):
    """Relation or objective split into the coefficient of every column and the rest."""

    coefficients: dict[int, Expression]
    rest: Expression


class LinearProgramsPass(BasePass):
    """Pass that lowers the linear program laws of the owners of a compilation and solves them in its functions."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'programs', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Lower the programs and make the functions of the problem solve them first."""
        compilation = self.compilation_
        groups = self._merged(self._bound_laws(compilation))
        problem = Compilation._required(compilation.problem, 'problem')
        if not groups:
            return
        states = LayoutSpec(tuple(Compilation._required(compilation.layout, 'layout').states))
        time = Compilation._required(compilation.layout, 'layout').time
        offsets = self._value_offsets(compilation)
        kernels = tuple(LinearProgramKernel(self._lowered(group, states, time, offsets)).prepare() for group in groups)
        compilation.problem = self._solving_problem(problem, kernels)

    @staticmethod
    def _count_values(bound: _BoundLawSpec, counts: dict[tuple[str, str], int]) -> None:
        """Count the hidden states of the values a law records."""
        law = bound.law
        counts[bound.owner, 'objective'] = counts.get((bound.owner, 'objective'), 0) + len(law.objectives)
        counts[bound.owner, 'dual'] = counts.get((bound.owner, 'dual'), 0) + (len(law.constraints) if law.duals else 0)

    @staticmethod
    def _value_offsets(compilation: Compilation) -> dict[int, tuple[int, int]]:
        """Return the position of the first objective value and first dual of every law, by the identity of the law."""
        counts: dict[tuple[str, str], int] = {}
        offsets = {}
        for bound in LinearProgramsPass._bound_laws(compilation, owned=False):
            offsets[id(bound.law)] = (counts.get((bound.owner, 'objective'), 0), counts.get((bound.owner, 'dual'), 0))
            LinearProgramsPass._count_values(bound, counts)
        return offsets

    @staticmethod
    def _bound_laws(compilation: Compilation, *, owned: bool = True) -> list[_BoundLawSpec]:
        """Return the linear program laws of the owners of a compilation, or of the whole system, with their owners."""
        laws: Laws = Compilation._required(compilation.programs, 'programs')
        found = [
            _BoundLawSpec(spec, law, CensusPass._law_owner(spec, law, set())) for spec, items in laws for law in items
        ]
        owners = compilation.owners
        return [item for item in found if not owned or owners is None or item.owner in owners]

    @staticmethod
    def _key_of(spec: KernelSpec, item: Expression) -> _Key | None:
        """Return what a symbol reads, the name of its state and its column, or `None` for no state."""
        reference = spec.references.get(item)
        return None if reference is None or not reference.is_state else (reference.entry.name, reference.column)

    @staticmethod
    def _variable_keys(bound: _BoundLawSpec) -> list[_Key]:
        """Return the states a law sets."""
        keys = [LinearProgramsPass._key_of(bound.spec, item) for item in bound.law._get_targets()]
        return [item for item in keys if item is not None]

    @staticmethod
    def _merged(bounds: Sequence[_BoundLawSpec]) -> list[list[_BoundLawSpec]]:
        """Return the laws of every program: those whose variables overlap, directly or through another law."""
        graph = nx.Graph()
        for position, bound in enumerate(bounds):
            graph.add_node(('law', position))
            graph.add_edges_from(
                (('law', position), ('state', key)) for key in LinearProgramsPass._variable_keys(bound)
            )
        components = [sorted(item[1] for item in part if item[0] == 'law') for part in nx.connected_components(graph)]
        return [[bounds[position] for position in part] for part in sorted(components)]

    @staticmethod
    def _columns_of(group: Sequence[_BoundLawSpec]) -> dict[_Key, int]:
        """Return the column of every variable of a program, in the order the laws declare them."""
        keys = [key for bound in group for key in LinearProgramsPass._variable_keys(bound)]
        return {key: position for position, key in enumerate(dict.fromkeys(keys))}

    @staticmethod
    def _symbols_in(spec: KernelSpec, expression: Expression, columns: Mapping[_Key, int]) -> dict[Expression, int]:
        """Return the column of every symbol of an expression that reads a variable of a program."""
        found = {}
        for item in expression.variables():
            key = LinearProgramsPass._key_of(spec, item)
            if key is not None and key in columns:
                found[item] = columns[key]
        return found

    @staticmethod
    def _linear_parts(spec: KernelSpec, expression: Expression, columns: Mapping[_Key, int]) -> _LinearPartsSpec:
        """Return the coefficient of every column of an expression and what is left when the variables are zero.

        The coefficients come from substitutions, since a condition in a bound has no derivative.
        """
        symbols = LinearProgramsPass._symbols_in(spec, expression, columns)
        zero = dict.fromkeys(symbols, 0)
        rest = expression.subs(zero)
        coefficients = {column: expression.subs({**zero, item: 1}) - rest for item, column in symbols.items()}
        return _LinearPartsSpec(coefficients, rest)

    @staticmethod
    def _form_problem(spec: KernelSpec, expression: Expression, columns: Mapping[_Key, int]) -> str | None:
        """Return what makes an expression not linear in the variables with coefficients free of states, or `None`."""
        parts = LinearProgramsPass._linear_parts(spec, expression, columns)
        symbols = LinearProgramsPass._symbols_in(spec, expression, columns)
        names = {column: str(item) for item, column in symbols.items()}
        linear = sum((parts.coefficients[column] * item for item, column in symbols.items()), parts.rest)
        if not (expression - linear).expand().is_trivial_zero():
            return f'it is not linear in the variables {", ".join(sorted(names.values()))}'
        for column, coefficient in parts.coefficients.items():
            states = [
                str(item) for item in coefficient.variables() if LinearProgramsPass._key_of(spec, item) is not None
            ]
            if states:
                return f'the coefficient of {names[column]} reads the states {", ".join(states)}'
        return None

    @staticmethod
    def _variable_issues(bound: _BoundLawSpec, algebraic: set[str]) -> list[IssueSpec]:
        """Return the variables of a law that are not algebraic states."""
        issues = []
        for item in bound.law._get_targets():
            reference = bound.spec.references.get(item)
            name = str(item) if reference is None else reference.entry.name
            if reference is None or name not in algebraic:
                message = f'The linear program law of {bound.spec.owner} sets {name}, which is not an algebraic state.'
                fix = f'Declare {name} with `StateSpec(algebraic=True)`.'
                issues.append(IssueSpec('linear-program-variable', bound.spec.owner, message, fix))
        return issues

    @staticmethod
    def _form_issues(group: Sequence[_BoundLawSpec]) -> list[IssueSpec]:
        """Return the relations and objectives of a program that are not linear in its variables."""
        columns = LinearProgramsPass._columns_of(group)
        fix = 'Move the term to the right side or to a bound, and keep every coefficient of a variable free of states.'
        issues = []
        for bound in group:
            spec = bound.spec
            for constraint in bound.law.constraints:
                problem = LinearProgramsPass._form_problem(spec, constraint.lhs() - constraint.rhs(), columns)
                if problem is not None:
                    message = f'The relation {constraint} of {spec.owner} cannot enter a linear program: {problem}.'
                    issues.append(IssueSpec('linear-program-form', spec.owner, message, fix))
            for objective in bound.law.objectives:
                linear = LinearProgramsPass._without_absolutes(objective)
                problem = LinearProgramsPass._form_problem(spec, linear, columns)
                if problem is None:
                    problem = LinearProgramsPass._absolute_problem(spec, objective, columns)
                if problem is not None:
                    message = f'The objective {objective.expression} of {spec.owner} is not usable: {problem}.'
                    issues.append(IssueSpec('linear-program-form', spec.owner, message, fix))
        return issues

    @staticmethod
    def _objective_issues(group: Sequence[_BoundLawSpec]) -> list[IssueSpec]:
        """Return the issue of a program with two laws that hold objectives."""
        holders = [bound.spec.owner for bound in group if bound.law.objectives]
        if len(holders) <= 1:
            return []
        message = f'The merged linear program laws of {" and ".join(holders)} both hold objectives.'
        fix = 'Keep the objectives of one program in one law.'
        return [IssueSpec('linear-program-objectives', holders[0], message, fix)]

    @staticmethod
    def _issues_of(compilation: Compilation, algebraic: set[str]) -> list[IssueSpec]:
        """Return the issues of the linear program laws of a system.

        Args:
            compilation:
                The compilation, which holds the programs.

            algebraic:
                The dotted names of the algebraic states of the system.

        Returns:
            issues:
                The issues, each with its path and fix.
        """
        bounds = LinearProgramsPass._bound_laws(compilation, owned=False)
        issues = [issue for bound in bounds for issue in LinearProgramsPass._variable_issues(bound, algebraic)]
        if issues:
            return issues
        for group in LinearProgramsPass._merged(bounds):
            issues += LinearProgramsPass._form_issues(group)
            issues += LinearProgramsPass._objective_issues(group)
        return issues

    @staticmethod
    def _without_absolutes(objective: LinearObjectiveSpec) -> Expression:
        """Return an objective with every absolute value replaced by a placeholder symbol."""
        expression = SR(objective.expression)
        return expression.subs(
            {
                call: symbol(f'_absolute_{position}')
                for position, call in enumerate(BasePass._calls_of(expression, ABSOLUTE))
            },
        )

    @staticmethod
    def _absolute_problem(
        spec: KernelSpec,
        objective: LinearObjectiveSpec,
        columns: Mapping[_Key, int],
    ) -> str | None:
        """Return what keeps the absolute values of an objective from a split into parts, or `None`."""
        expression = SR(objective.expression)
        linear = LinearProgramsPass._without_absolutes(objective)
        for position, call in enumerate(BasePass._calls_of(expression, ABSOLUTE)):
            argument = call.operands()[0]
            if LinearProgramsPass._key_of(spec, argument) not in columns:
                return f'the absolute value {call} does not hold a single variable'
            coefficient = linear.diff(symbol(f'_absolute_{position}'))
            if not coefficient.is_numeric():
                return f'the coefficient of {call} is not a number'
            sign = 1.0 if objective.sense == 'min' else -1.0
            if sign * float(coefficient) < 0:
                return f'it {objective.sense}imises the absolute value {call} with the sign that is not convex'
        return None

    @staticmethod
    def _printed_values(
        expressions: Sequence[Expression],
        references: Mapping[Expression, ReferenceSpec],
        time: Expression,
    ) -> tuple[Callable[..., list[npt.ArrayLike]] | None, tuple[ReferenceSpec, ...]]:
        """Return the compiled function of the values of a program and the references it reads."""
        if not expressions:
            return None, ()
        arguments = sorted(symbols_of(list(expressions)) - {time}, key=str)
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run(list(expressions)).code_
        if not callable(function):
            error_msg = 'The printer returns no function of the bounds of a linear program.'
            raise TypeError(error_msg)
        return function, tuple(references[item] for item in arguments)

    @staticmethod
    def _lowered(
        group: Sequence[_BoundLawSpec],
        states: LayoutSpec,
        time: Expression,
        offsets: Mapping[int, tuple[int, int]],
    ) -> LinearProgramSpec:
        """Return the program of laws that share variables, with its constant matrix and its compiled values."""
        return _Lowering(group, states, time, offsets).lowered_program()

    @staticmethod
    def _solving_problem(problem: ProblemSpec, kernels: tuple[LinearProgramKernel, ...]) -> ProblemSpec:
        """Return a problem whose functions solve the programs first, without a compiled shortcut around them."""

        def solved(t: float, x: Array, p: Array) -> Array:
            state = x
            for kernel in kernels:
                state = kernel.run(RunStateSpec(t, state, p)).values_
            return state

        drift, update, algebraic = problem.drift, problem.update, problem.algebraic

        def solving_drift(t: float, x: Array, p: Array) -> Array:
            if drift is None:
                return np.zeros_like(x)
            return drift(t, solved(t, x, p), p)

        def solving_update(t: float, x: Array, p: Array, *drawn: object) -> Array:
            state = solved(t, x, p)
            return state if update is None else update(t, state, p, *drawn)

        def solving_algebraic(t: float, x: Array, p: Array) -> Array:
            state = solved(t, x, p)
            return state if algebraic is None else algebraic(t, state, p)

        return replace(
            problem,
            drift=solving_drift if drift is not None else None,
            update=solving_update if update is not None else None,
            algebraic=solving_algebraic,
            programs=kernels,
            jacobian=None,
            accelerated=None,
            has_callables=True,
        )


class _Lowering:
    """Lowering of one program: a row per relation in several variables, a bound per relation in one."""

    def __init__(
        self,
        group: Sequence[_BoundLawSpec],
        states: LayoutSpec,
        time: Expression,
        offsets: Mapping[int, tuple[int, int]],
    ) -> None:
        self.group = group
        self.states = states
        self.time = time
        self.offsets = offsets

    def _indices_of(self, name: str) -> Indices:
        """Return the flat indices of a hidden state."""
        entry = self.states.entry(name)
        return entry.start + np.arange(entry.size)

    def _value(self, expression: Expression) -> int:
        """Return the position of a compiled value."""
        self._expressions.append(expression)
        return len(self._expressions) - 1

    def _set_bound(self, column: int, sense: object, coefficient: float, side: Expression) -> tuple[int, ...]:
        """Add a relation in one variable with a number as its coefficient to the bounds, and return their kinds."""
        kinds: dict[object, tuple[int, ...]] = {
            operator.eq: (LinearProgramBound.COLUMN_LOWER, LinearProgramBound.COLUMN_UPPER),
            operator.le: (LinearProgramBound.COLUMN_UPPER,) if coefficient > 0 else (LinearProgramBound.COLUMN_LOWER,),
            operator.ge: (LinearProgramBound.COLUMN_LOWER,) if coefficient > 0 else (LinearProgramBound.COLUMN_UPPER,),
        }
        for kind in kinds[sense]:
            if side.is_numeric():
                value = float(side) / coefficient
                if kind == LinearProgramBound.COLUMN_LOWER:
                    self._lower[column] = max(self._lower[column], value)
                else:
                    self._upper[column] = min(self._upper[column], value)
            else:
                self._sides.append(LinearProgramSideSpec(kind, column, self._value(side), coefficient))
        return kinds[sense]

    def _add_row(self, coefficients: Mapping[int, Expression], sense: object, side: Expression, text: str) -> int:
        """Add a row of a relation in several variables, or with a coefficient that reads parameters."""
        row = len(self._row_lower)
        self._constraints.append(text)
        kinds: dict[object, tuple[int, ...]] = {
            operator.eq: (LinearProgramBound.ROW_LOWER, LinearProgramBound.ROW_UPPER),
            operator.le: (LinearProgramBound.ROW_UPPER,),
            operator.ge: (LinearProgramBound.ROW_LOWER,),
        }
        self._row_lower.append(-np.inf)
        self._row_upper.append(np.inf)
        for kind in kinds[sense]:
            if side.is_numeric():
                (self._row_lower if kind == LinearProgramBound.ROW_LOWER else self._row_upper)[row] = float(side)
            else:
                self._sides.append(LinearProgramSideSpec(kind, row, self._value(side)))
        for column, coefficient in coefficients.items():
            if coefficient.is_numeric():
                self._triplets.append((row, column, float(coefficient)))
            else:
                self._entries.append(LinearProgramEntrySpec(row, column, self._value(coefficient)))
        return row

    def _add_relation(self, bound: _BoundLawSpec, relation: Expression, position: int) -> None:
        """Add a relation of a law as a bound or a row, and the place of its dual when the law records it."""
        spec = bound.spec
        parts = LinearProgramsPass._linear_parts(spec, relation.lhs() - relation.rhs(), self._columns)
        side = -parts.rest
        sense = relation.operator()
        text = str(relation)
        self._relations.append(text)
        coefficients = parts.coefficients
        only = next(iter(coefficients.values())) if len(coefficients) == 1 else None
        column, row, scale = -1, -1, 1.0
        kinds: tuple[int, ...] = ()
        if only is not None and only.is_numeric() and not only.is_zero():
            column = next(iter(coefficients))
            scale = float(only)
            kinds = self._set_bound(column, sense, scale, side)
        elif coefficients:
            row = self._add_row(coefficients, sense, side, text)
        if bound.law.duals:
            index = self.offsets[id(bound.law)][1] + position
            self._duals.append(
                LinearProgramDualSpec(
                    self._indices_of(f'{bound.owner}.dual_{index}'),
                    text,
                    row=row,
                    column=column,
                    kinds=kinds,
                    scale=1.0 / scale,
                ),
            )

    def _objective_row(self, bound: _BoundLawSpec, objective: LinearObjectiveSpec, position: int) -> None:
        """Add the costs of an objective, with parts for the absolute values of its variables."""
        spec = bound.spec
        expression = SR(objective.expression)
        linear = LinearProgramsPass._without_absolutes(objective)
        sign = 1.0 if objective.sense == 'min' else -1.0
        parts = LinearProgramsPass._linear_parts(spec, linear, self._columns)
        costs: dict[int, float] = {}
        for column, coefficient in parts.coefficients.items():
            if coefficient.is_numeric():
                costs[column] = sign * float(coefficient)
            else:
                self._cost_entries.append(LinearProgramEntrySpec(position, column, self._value(coefficient)))
        rest = parts.rest
        for index, call in enumerate(BasePass._calls_of(expression, ABSOLUTE)):
            placeholder = symbol(f'_absolute_{index}')
            coefficient = float(linear.diff(placeholder))
            rest = rest.subs({placeholder: 0})
            key = LinearProgramsPass._key_of(spec, call.operands()[0])
            column = self._columns[key] if key is not None else -1
            place = self._parts.setdefault(column, len(self._parts))
            costs[-1 - place] = sign * coefficient
        self._costs.append(costs)
        self._signs.append(sign)
        self._offsets.append(float(rest) if rest.is_numeric() else 0.0)

    def _cost_matrix(self) -> Array:
        """Return the costs of every objective, with the parts after the variables."""
        size, count = len(self._columns), len(self._parts)
        matrix = np.zeros((len(self._costs), size + 2 * count))
        for row, costs in enumerate(self._costs):
            for column, value in costs.items():
                if column >= 0:
                    matrix[row, column] = value
                else:
                    matrix[row, [size - 1 - column, size + count - 1 - column]] = value
        return matrix

    def _set_empty(self) -> None:
        """Start the lists of the program."""
        size = len(self._columns)
        self._expressions: list[Expression] = []
        self._lower, self._upper = np.full(size, -np.inf), np.full(size, np.inf)
        self._sides: list[LinearProgramSideSpec] = []
        self._entries: list[LinearProgramEntrySpec] = []
        self._cost_entries: list[LinearProgramEntrySpec] = []
        self._constraints: list[str] = []
        self._relations: list[str] = []
        self._row_lower: list[float] = []
        self._row_upper: list[float] = []
        self._triplets: list[tuple[int, int, float]] = []
        self._duals: list[LinearProgramDualSpec] = []
        self._costs: list[dict[int, float]] = []
        self._signs: list[float] = []
        self._offsets: list[float] = []
        self._parts: dict[int, int] = {}

    def lowered_program(self) -> LinearProgramSpec:
        """Return the lowered program."""
        group = self.group
        self._columns = LinearProgramsPass._columns_of(group)
        self._set_empty()
        names, indices = self._names_and_indices()
        activity = [self._value(SR(bound.law.active)) for bound in group if bound.law.active is not None]
        for bound in group:
            for position, relation in enumerate(bound.law.constraints):
                self._add_relation(bound, relation, position)
        holder = next((bound for bound in group if bound.law.objectives), group[0])
        objectives = []
        for position, objective in enumerate(holder.law.objectives):
            self._objective_row(holder, objective, position)
            index = self.offsets[id(holder.law)][0] + position
            objectives.append(self._indices_of(f'{holder.owner}.objective_{index}'))
        rows = len(self._row_lower)
        triplets = np.asarray(self._triplets, dtype=float).reshape(-1, 3)
        matrix = sparse.csr_array(
            (triplets[:, 2], (triplets[:, 0].astype(np.intp), triplets[:, 1].astype(np.intp))),
            shape=(rows, len(self._columns)),
        )
        references = {key: value for bound in group for key, value in bound.spec.references.items()}
        function, arguments = LinearProgramsPass._printed_values(self._expressions, references, self.time)
        return LinearProgramSpec(
            path=holder.owner,
            names=names,
            variables=indices,
            constraints=tuple(self._constraints),
            relations=tuple(self._relations),
            matrix=matrix,
            lower=self._lower,
            upper=self._upper,
            row_lower=np.asarray(self._row_lower, dtype=float),
            row_upper=np.asarray(self._row_upper, dtype=float),
            sides=tuple(self._sides),
            entries=tuple(self._entries),
            function=function,
            arguments=arguments,
            members=max(bound.spec.rows for bound in group),
            activity=tuple(activity),
            costs=self._cost_matrix(),
            cost_entries=tuple(self._cost_entries),
            signs=tuple(self._signs),
            offsets=tuple(self._offsets),
            parts=tuple(self._parts),
            objectives=tuple(objectives),
            duals=tuple(self._duals),
            infeasible=holder.law.infeasible,
        )

    def _names_and_indices(self) -> tuple[tuple[str, ...], tuple[Indices, ...]]:
        """Return the name and the flat indices of every variable, the component in the name of a vector."""
        names, indices = [], []
        for name, column in self._columns:
            entry = self.states.entry(name)
            names.append(entry.generic_names[column] if entry.columns > 1 else name)
            indices.append(entry.start + np.arange(entry.rows) * entry.columns + column)
        return tuple(names), tuple(indices)
