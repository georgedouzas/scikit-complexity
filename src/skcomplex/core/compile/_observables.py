"""Compile the observables of a system tree into one evaluator called at every frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError, components_of, symbol, symbols_of
from ..model import Observable, Symbols, System, reduce_values
from ._printing import numpy_function
from ._problem import Problem, Vector

type AgentsOf = Callable[[str], list[Any] | None]
type Getter = Callable[[float, Vector, Vector, Mapping[str, Any]], Any]


def _from_done(name: str) -> Callable[[Vector, Vector, Mapping[str, Any]], Any]:
    """Return a reader of an observable computed earlier in the same frame."""
    return lambda x, p, done: done[name]


def _from_members(read: Callable[[Vector, Vector], Any]) -> Callable[[Vector, Vector, Mapping[str, Any]], Any]:
    """Return a reader of one member variable over a group."""
    return lambda x, p, done: read(x, p)


def _from_vector(is_state: bool, indices: Any) -> Callable[[Vector, Vector], Any]:  # noqa: ANN401
    """Return a reader of components of the state or the parameter vector."""
    return lambda x, p: (x if is_state else p)[indices]


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

    def _scalar_sources(self) -> dict[Expression, tuple[str, int]]:
        """Return the vector and flat index of every symbol of a single element or interaction."""
        sources: dict[Expression, tuple[str, int]] = {}
        for vector, layout in (('x', self.problem.states), ('p', self.problem.parameters)):
            for entry in layout:
                if entry.members is not None or entry.name not in self.problem.symbols:
                    continue
                for column, item in enumerate(components_of(self.problem.symbols[entry.name])):
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
        expression = SR(expression)
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
        members: dict[Expression, Callable[[Vector, Vector], Any]] = {}
        for name in names.values():
            entry = self._entry(name)
            is_state = name in self.problem.states
            for column, item in enumerate(components_of(self.problem.symbols[name])):
                indices = entry.start + np.arange(size) * entry.columns + column
                members[item] = _from_vector(is_state, indices)
        namespace = Symbols(
            **{variable: self.problem.symbols[name] for variable, name in names.items()},
            t=self.problem.time,
        )
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        evaluate = self._compile(path, expression, {}, members)

        def read_expression(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            values = np.broadcast_to(evaluate(t, x, p, done), (size,))
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
