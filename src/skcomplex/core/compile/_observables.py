"""Compile the observables of a system tree into one evaluator called at every frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, I

from ..base import Domain, Expression, RoleError, ValidationError, real_components_of, symbol, symbols_of
from ..model import Observable, Reduction, Symbols, System, reduce_values, reductions_in
from ._complex import joined_complex
from ._layout import Entry
from ._printing import numpy_function
from ._problem import Problem, Vector
from ._quantum import QuantumProblem, quantum_getters
from ._reductions import CompiledReduction, Read, ReductionPass

type AgentsOf = Callable[[str], list[Any] | None]
type Getter = Callable[[float, Vector, Vector, Mapping[str, Any]], Any]


def _from_done(name: str) -> Callable[[Vector, Vector, Mapping[str, Any]], Any]:
    """Return a reader of an observable computed earlier in the same frame."""
    return lambda x, p, done: done[name]


def _from_members(read: Callable[[Vector, Vector], Any]) -> Callable[[Vector, Vector, Mapping[str, Any]], Any]:
    """Return a reader of one member variable over a group."""
    return lambda x, p, done: read(x, p)


def _read_of(read: Read) -> Callable[[Vector, Vector], Any]:
    """Return a reader of the values a reduction read stands for, from the state or the extended parameters."""
    return _from_vector(read.is_state, read.index)


def _from_vector(is_state: bool, indices: Any) -> Callable[[Vector, Vector], Any]:  # noqa: ANN401
    """Return a reader of components of the state or the parameter vector."""
    return lambda x, p: (x if is_state else p)[indices]


def _renewal_counter(counts: npt.NDArray[np.intp]) -> Getter:
    """Return the getter of the number of renewals of a renewal law over all its members."""

    def count_renewals(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> float:
        del t, p, done
        return float(np.sum(x[counts]))

    return count_renewals


def _join(scope: str, path: str) -> str:
    """Return a path relative to a system as a path from the root."""
    return f'{scope}/{path}' if scope else path


def observables_of(system: System, scope: str = '') -> Iterator[tuple[str, Observable]]:
    """Yield the path of every system in a tree that declares observables, with each observable.

    Args:
        system:
            The root system.

        scope:
            The path of the system, empty for the root.

    Yields:
        observable:
            The path of the declaring system and the observable.
    """
    for observable in getattr(system, 'observables', ()):
        yield scope, observable
    for child in system.systems:
        yield from observables_of(child, _join(scope, child.label))


def _tree(items: Mapping[str, Any], **root: Any) -> Symbols:  # noqa: ANN401
    """Return nested namespaces from paths such as `forest/foxes.n` mapped to values, with extra values at the root."""
    nested: dict[str, Any] = {}
    for path, value in items.items():
        *systems, last = path.split('/')
        node = nested
        for name in systems:
            node = node.setdefault(name, {})
        element, _, variable = last.partition('.')
        node.setdefault(element, {})[variable] = value

    def build(node: dict[str, Any]) -> Symbols:
        return Symbols(**{key: build(value) if isinstance(value, dict) else value for key, value in node.items()})

    return Symbols(**{key: build(value) for key, value in nested.items()}, **root)


@dataclass
class _MemberReductions:
    """Reductions in the member expression of an observable, compiled with the reduction lowering of laws."""

    path: str
    group: str
    size: int
    reads: dict[Expression, Read]
    time: Expression
    parameters: int
    found: dict[str, CompiledReduction] = field(default_factory=dict)

    @property
    def extent(self) -> int:
        """Size of the parameter vector extended by the values of the reductions."""
        return max((item.entry.start + item.entry.size for item in self.found.values()), default=self.parameters)

    def _check_over(self, reduction: Reduction) -> None:
        """Raise an error when a reduction reduces over a group other than the observable's group."""
        read = self.reads.get(reduction.over) if reduction.over is not None else None
        if reduction.over is None or (read is not None and read.per_member):
            return
        error_msg = (
            f'The reduction {reduction.call} of observable {self.path} reads a group other than {self.group}. A '
            'reduction in the member expression of an observable reduces over the observable\'s group, so write one '
            'observable per group.'
        )
        raise RoleError(error_msg)

    def _read(self, item: Expression) -> Read:
        """Return how the reduction pass reads one symbol of a reduction."""
        if item not in self.reads:
            error_msg = (
                f'A reduction of observable {self.path} reads {item}, which is not a variable of a member of '
                f'{self.group}, of a single element, or another reduction. Reductions read the namespace the '
                'expression receives.'
            )
            raise ValidationError(error_msg)
        return self.reads[item]

    def _weights(self, reduction: Reduction) -> Any:  # noqa: ANN401
        """Return the weights of a reduction as a CSR array after checking their shape."""
        if reduction.weights is None:
            return None
        shape = tuple(reduction.weights.shape)
        if shape != (self.size, self.size):
            error_msg = (
                f'The weights of the reduction {reduction.call} of observable {self.path} have the shape {shape}, but '
                f'the reduction needs the shape {(self.size, self.size)}, one row and one column per member of '
                f'{self.group}.'
            )
            raise ValidationError(error_msg)
        return scipy.sparse.csr_array(reduction.weights, dtype=float)

    def _placeholder(self, reduction: Reduction) -> Expression:
        """Return the placeholder of a reduction, compiling it the first time it is found."""
        key = str(reduction.call)
        if key in self.found:
            return self.found[key].symbol
        if SR(reduction.argument).has(I):
            error_msg = (
                f'The reduction {reduction.call} of observable {self.path} reduces a complex expression. Reduce its '
                'real or imaginary part.'
            )
            raise ValidationError(error_msg)
        self._check_over(reduction)
        argument = SR(self.lower(reduction.argument))
        condition = None if reduction.condition is None else SR(self.lower(reduction.condition))
        symbols = symbols_of([argument, *([condition] if condition is not None else [])]) - {self.time}
        weights = self._weights(reduction)
        index = len(self.found)
        shape = (self.size,) if weights is not None else ()
        members = self.size if weights is not None else None
        entry = Entry(f'__observed_{index}', self.extent, shape, Domain.REAL, members)
        placeholder = symbol(f'_observed_{index}')
        reads = {item: self._read(item) for item in sorted(symbols, key=str)}
        self.found[key] = CompiledReduction(
            placeholder,
            reduction.kind,
            argument,
            condition,
            weights,
            entry,
            self.size,
            self.group,
            reads,
            self.time,
        )
        indices = entry.start + np.arange(self.size) if weights is not None else entry.start
        self.reads[placeholder] = Read(False, indices, entry)
        return placeholder

    def lower(self, expression: Any) -> Any:  # noqa: ANN401
        """Return an expression with every reduction replaced by the placeholder of its value."""
        if not isinstance(expression, Expression):
            return expression
        found = reductions_in(expression)
        placeholders = {item.call: self._placeholder(item) for item in found}
        lowered = expression
        for call in sorted(placeholders, key=lambda item: -len(str(item))):
            lowered = lowered.subs({call: placeholders[call]})
        return lowered


@dataclass
class ObservableEvaluator:
    """Evaluator of every observable of a problem from a state vector, a parameter vector, and the agents.

    Args:
        problem:
            The compiled problem.

    Examples:
        >>> from skcomplex.core.compile import ObservableEvaluator, compile_system
        >>> from skcomplex.core.model import Observable, System
        >>> from skcomplex.models.biology.ecology import Population
        >>> system = System(
        ...     elements=[Population.group('herd', 2, n=[2.0, 6.0])],
        ...     observables=[Observable('total', group='herd', reduce='sum', variable='n')],
        ... )
        >>> problem = compile_system(system)
        >>> ObservableEvaluator(problem).evaluate(0.0, problem.initial_state, problem.initial_parameters)
        {'total': 8.0}
    """

    problem: Problem

    def __post_init__(self) -> None:
        """Compile every observable into a getter, in declaration order."""
        self._scalars = self._scalar_sources()
        self.getters_: dict[str, Getter] = {}
        system = self.problem.system
        if isinstance(self.problem, QuantumProblem):
            self.getters_ = quantum_getters(self.problem, observables_of(system) if system is not None else ())
            return
        for scope, observable in observables_of(system) if system is not None else ():
            path = _join(scope, observable.label)
            if path in self.getters_:
                error_msg = f'Two observables have the path {path}. Give observables of one system distinct labels.'
                raise ValidationError(error_msg)
            if observable.group is None and observable.expression is None:
                error_msg = f'Observable {path} has neither an expression nor a group. Give one of them.'
                raise ValidationError(error_msg)
            if observable.group is None:
                self.getters_[path] = self._expression_getter(scope, path, observable)
            else:
                self.getters_[path] = self._reduction_getter(scope, path, observable)
        problems = (self.problem, *(partition.problem for partition in self.problem.partitions))
        for renewal in (renewal for problem in problems for renewal in problem.renewals):
            self.getters_[f'{renewal.label}.renewals'] = _renewal_counter(renewal.counts)

    def _scalar_sources(self) -> dict[Expression, tuple[str, int]]:
        """Return the vector and flat index of every symbol of a single element or interaction."""
        sources: dict[Expression, tuple[str, int]] = {}
        for vector, layout in (('x', self.problem.states), ('p', self.problem.parameters)):
            for entry in layout:
                if entry.members is not None or entry.name not in self.problem.symbols:
                    continue
                for column, item in enumerate(real_components_of(self.problem.symbols[entry.name])):
                    sources[item] = (vector, entry.start + column)
        return sources

    def _compile(
        self,
        path: str,
        expression: Any,  # noqa: ANN401
        observables: Mapping[Expression, str],
        members: Mapping[Expression, Callable[[Vector, Vector], Any]],
    ) -> Getter:
        """Return a getter that evaluates an expression from scalars, observables, and member arrays."""
        expression = joined_complex(expression, f'observable {path}')
        arguments = sorted(symbols_of([expression]) - {self.problem.time}, key=str)
        readers: list[Callable[[Vector, Vector, Mapping[str, Any]], Any]] = []
        for item in arguments:
            if item in observables:
                readers.append(_from_done(observables[item]))
            elif item in members:
                readers.append(_from_members(members[item]))
            elif item in self._scalars:
                vector, index = self._scalars[item]
                readers.append(_from_members(_from_vector(vector == 'x', index)))
            else:
                error_msg = (
                    f'Observable {path} uses the symbol {item}, which is neither a variable of a single element or '
                    'interaction nor an earlier observable. Use the namespace that the expression receives.'
                )
                raise ValidationError(error_msg)
        function = numpy_function(self.problem.time, arguments, [expression])

        def get(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            value = function(t, *(read(x, p, done) for read in readers))[0]
            value = np.real_if_close(value)
            return float(value) if np.ndim(value) == 0 and np.isrealobj(value) else value

        return get

    def _expression_getter(self, scope: str, path: str, observable: Observable) -> Getter:
        """Return the getter of an expression of the system's symbols."""
        prefix = f'{scope}/' if scope else ''
        items: dict[str, Any] = {
            name[len(prefix) :]: self.problem.symbols[name]
            for name in self.problem.symbols
            if name.startswith(prefix) and (entry := self._entry(name)) is not None and entry.members is None
        }
        earlier = {
            other: symbol(f'_observable_{index}')
            for index, other in enumerate(self.getters_)
            if other.startswith(prefix) and '/' not in other[len(prefix) :]
        }
        namespace = _tree(items, t=self.problem.time)
        for other, item in earlier.items():
            setattr(namespace, other[len(prefix) :], item)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        return self._compile(path, expression, {item: other for other, item in earlier.items()}, {})

    def _entry(self, name: str) -> Any:  # noqa: ANN401
        """Return the entry of a state or parameter by name, or `None`."""
        for layout in (self.problem.states, self.problem.parameters):
            if name in layout:
                return layout.entry(name)
        return None

    def _reduction_getter(self, scope: str, path: str, observable: Observable) -> Getter:
        """Return the getter of a reduction over a group of elements or agents."""
        group = _join(scope, observable.group or '')
        element = next((item for item in self.problem.elements if item.path == group), None)
        if element is None:
            return self._agent_getter(group, path, observable)
        names = {**element.states, **element.parameters}
        if observable.variable is not None:
            if observable.variable not in names:
                error_msg = f'Observable {path} reduces {observable.variable}, which {group} does not declare.'
                raise ValidationError(error_msg)
            name = names[observable.variable]
            is_state = name in self.problem.states
            layout = self.problem.states if is_state else self.problem.parameters

            def read_variable(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
                del t, done
                values = np.asarray(layout.read(x if is_state else p, name))
                return reduce_values(values if element.members is not None else values[np.newaxis], observable.reduce)

            return read_variable
        size = element.members or 1
        if observable.expression is None:
            if observable.reduce != 'count':
                error_msg = f'Observable {path} reduces {group} with {observable.reduce!r} but names no variable.'
                raise ValidationError(error_msg)
            return lambda t, x, p, done: float(size)
        members: dict[Expression, Read] = {}
        for name in names.values():
            entry = self._entry(name)
            is_state = name in self.problem.states
            for column, item in enumerate(real_components_of(self.problem.symbols[name])):
                indices = (entry.start + np.arange(size) * entry.columns + column).astype(np.intp)
                members[item] = Read(is_state, indices, entry, column, entry.columns)
        namespace = Symbols(
            **{variable: self.problem.symbols[name] for variable, name in names.items()},
            t=self.problem.time,
        )
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        scalars = {item: Read(vector == 'x', index) for item, (vector, index) in self._scalars.items()}
        lowering = _MemberReductions(
            path,
            group,
            size,
            {**scalars, **members},
            self.problem.time,
            len(self.problem.initial_parameters),
        )
        expression = lowering.lower(expression)
        readers = {item: _read_of(read) for item, read in lowering.reads.items() if item not in scalars}
        evaluate = self._compile(path, expression, {}, readers)
        reductions = ReductionPass(tuple(lowering.found.values()), lowering.parameters)

        def read_expression(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            values = np.broadcast_to(evaluate(t, x, reductions.extend(t, x, p), done), (size,))
            return reduce_values(values, observable.reduce)

        return read_expression

    def _agent_getter(self, group: str, path: str, observable: Observable) -> Getter:
        """Return the getter of a reduction over an agent group, which the evaluation's agents give."""
        if not any(spec.path == group for spec in self.problem.agents) and not self.problem.partitions:
            error_msg = f'Observable {path} reduces {group}, which is neither an element group nor an agent group.'
            raise ValidationError(error_msg)

        def read_agents(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            del t, x, p
            agents = done['__agents__'](group) or []
            if observable.variable is not None:
                values: Any = [getattr(agent, observable.variable) for agent in agents]
            elif callable(observable.expression):
                values = [observable.expression(agent) for agent in agents]
            else:
                values = agents
            return reduce_values(values, observable.reduce)

        return read_agents

    def evaluate(self, t: float, x: Vector, p: Vector, agents_of: AgentsOf | None = None) -> dict[str, Any]:
        """Return the value of every observable by path.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

            agents_of:
                A function that returns the agents of an agent group by path, or `None` without agents.

        Returns:
            values:
                A mapping from observable path to its value.
        """
        done: dict[str, Any] = {'__agents__': agents_of or (lambda group: None)}
        for path, get in self.getters_.items():
            done[path] = get(t, x, p, done)
        del done['__agents__']
        return done
