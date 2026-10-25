"""Compile the observables of a system tree into one evaluator called at every frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, I

from ...core.base import (
    BaseProblem,
    Entry,
    Expression,
    Form,
    NumericDomain,
    RoleError,
    SymbolicVector,
    ValidationError,
    VariableKind,
    Vector,
    declarations_of,
    symbol,
    symbols_of,
    vector,
)
from ...core.components import Aggregate
from ...core.laws import Reduction, Symbols, count, maximum, mean, minimum, reductions_in, total, variance
from ...core.problems import Getter, Problem, ReductionSchedule, Reference, Source, numpy_function
from ...core.problems._problem import real_components_of
from ...core.problems._quantities import _Recorded
from ...core.problems._reads import Read
from ...core.problems._reductions import CompiledReduction
from ...core.system import Observable, System, reduce_values
from ._complex import joined_complex
from ._declarations import Block, Declarations
from ._specs import Observed
from ._tree import Node

type AgentsOf = Callable[[str], list[Any] | None]

REDUCERS: dict[str, Callable[..., Expression]] = {
    'sum': total,
    'mean': mean,
    'max': maximum,
    'min': minimum,
    'var': variance,
}


type Declared = dict[str, tuple[str, Observable]]


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


def _with_observables(system: System, observables: Sequence[Observable]) -> System:
    """Return a copy of a system tree that declares observables in the systems their labels name."""
    if not observables:
        return system
    by_scope: dict[str, list[Observable]] = {}
    for observable in observables:
        scope, _, label = observable.label.rpartition('/')
        local = copy.copy(observable)
        local.label = label
        by_scope.setdefault(scope, []).append(local)
    paths = {''} | set(_system_paths(system))
    unknown = sorted(set(by_scope) - paths)
    if unknown:
        error_msg = (
            f'The observables name the nested systems {", ".join(unknown)}, which do not exist. Label an observable '
            'by its path from the root, such as `village/mean_wealth`.'
        )
        raise ValidationError(error_msg)
    return _attached(system, by_scope, '')


def _system_paths(system: System, scope: str = '') -> Iterator[str]:
    """Yield the path of every nested system."""
    for child in system.systems:
        path = _join(scope, child.label)
        yield path
        yield from _system_paths(child, path)


def _attached(system: System, by_scope: dict[str, list[Observable]], scope: str) -> System:
    """Return a copy of a system that holds the observables of its scope, with its nested systems attached."""
    copied = copy.copy(system)
    copied.components = [
        _attached(component, by_scope, _join(scope, component.label)) if isinstance(component, System) else component
        for component in system.components
    ]
    copied.observables = [*getattr(system, 'observables', ()), *by_scope.get(scope, ())]
    return copied


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


def _is_lowerable(scope: str, observable: Observable, nodes: dict[str, Node]) -> bool:
    """Check whether laws can read an observable at the current state, as an expression or a symbolic reduction."""
    if observable.group is None:
        return True
    node = nodes.get(_join(scope, observable.group))
    return node is not None and not node.is_agent and isinstance(observable.reduce, str)


def _source_of(block: Block) -> Source:
    """Return how laws read the variables of a block from outside it, per member for a group."""
    return Source.MEMBER if block.node.members is not None else Source.SINGLE


def _reduced_value(path: str, observable: Observable, block: Block) -> object:
    """Return an observable that reduces a group as a reduction over the group's member symbols."""
    namespace, _ = block.namespace(_source_of(block))
    if observable.reduce == 'count':
        return count(over=namespace)
    if observable.variable is not None:
        if observable.variable not in block.symbols:
            error_msg = f'Observable {path} reduces {observable.variable}, which {block.node.path} does not declare.'
            raise ValidationError(error_msg)
        value: Any = block.symbols[observable.variable]
    elif observable.expression is None:
        error_msg = f'Observable {path} reduces {block.node.path} with {observable.reduce!r} but names no variable.'
        raise ValidationError(error_msg)
    else:
        value = observable.expression(namespace) if callable(observable.expression) else observable.expression
    reducer = REDUCERS[str(observable.reduce)]
    if isinstance(value, SymbolicVector):
        return vector(reducer(item, over=namespace) for item in value)
    return reducer(value, over=namespace)


def _role_paths(system: System, scope: str = '') -> set[str]:
    """Return the paths from the root of everything that roles of a system tree bind to."""
    paths = set()
    for interaction in system.interactions:
        for name in declarations_of(interaction, VariableKind.ROLE):
            value = getattr(interaction, name)
            labels = [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else []
            paths |= {f'{scope}/{label}' if scope else label for label in labels if isinstance(label, str)}
    for child in system.systems:
        paths |= _role_paths(child, f'{scope}/{child.label}' if scope else child.label)
    return paths


def _readers_of(problem: Problem, path: str) -> set[str]:
    """Return the paths of the elements and groups that interactions reading an observable bind their other roles to."""
    readers: set[str] = set()
    for interaction in problem.interactions:
        bound = [
            item for value in interaction.roles.values() for item in ([value] if isinstance(value, str) else value)
        ]
        if path in bound:
            readers |= set(bound) - {path}
    return readers


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


def lowerable_observables(system: System, nodes: list[Node]) -> Declared:
    """Return every observable of a system tree that laws can read at the current state, by path.

    An expression of a system's symbols is lowerable when every observable declared before it in its system is.
    """
    by_path = {node.path: node for node in nodes}
    lowerable: Declared = {}
    blocked: set[str] = set()
    for scope, observable in observables_of(system):
        if scope in blocked or not _is_lowerable(scope, observable, by_path):
            blocked.add(scope)
            continue
        lowerable[_join(scope, observable.label)] = (scope, observable)
    return lowerable


def with_held_observables(system: System, read: set[str], scope: str = '') -> tuple[System, list[str]]:
    """Return a system tree in which every observable in `read` is also an aggregate, and their paths."""
    held: list[str] = []
    aggregates = []
    for observable in getattr(system, 'observables', ()):
        path = f'{scope}/{observable.label}' if scope else observable.label
        if path not in read:
            continue
        if observable.group is None or observable.expression is not None:
            error_msg = (
                f'A role binds to the observable {path}, which reads agents through an expression or an earlier '
                'observable. Laws read observables of agents only when they reduce a variable or count the agents. '
                'Declare it with `group` and `variable`.'
            )
            raise ValidationError(error_msg)
        aggregates.append(
            Aggregate(observable.label, group=observable.group, reduce=observable.reduce, variable=observable.variable),
        )
        held.append(path)
    children = []
    for child in system.systems:
        new_child, child_held = with_held_observables(child, read, f'{scope}/{child.label}' if scope else child.label)
        children.append(new_child)
        held += child_held
    if not held:
        return system, held
    copied = copy.copy(system)
    copied.components = [*system.elements, *aggregates, *children]
    return copied, held


def held_observables(system: System, nodes: list[Node]) -> set[str]:
    """Return the paths of the observables that roles bind to and laws read only at synchronisation points.

    Args:
        system:
            The system, with the observables of the run.

        nodes:
            The nodes of the system.

    Returns:
        paths:
            The paths of the held observables.
    """
    lowerable = lowerable_observables(system, nodes)
    declared = {_join(scope, observable.label) for scope, observable in observables_of(system)}
    return (_role_paths(system) & declared) - set(lowerable)


def check_held_reads(problem: Problem, held: list[str]) -> None:
    """Raise an error when a law reads an observable that cannot be lowered from the partition that computes it."""
    groups = {
        _join(scope, observable.label): _join(scope, observable.group or '')
        for scope, observable in observables_of(problem.system)  # type: ignore[arg-type]
    }
    for path in held:
        readers = _readers_of(problem, path)
        shared = [item.name for item in problem.partitions if groups[path] in item.owners and readers & item.owners]
        if problem.form is not Form.COUPLED or shared:
            error_msg = (
                f'A law reads the observable {path} in the partition that computes it, but the observable is a Python '
                'function of the values or reduces agents, which laws cannot read at the current state. Write it '
                'symbolically with a variable or an expression of the members, or move the reading law into another '
                'subsystem of a different form.'
            )
            raise ValidationError(error_msg)


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
        weights = reduction.weights
        if np.ndim(weights) == 1 and len(weights) == self.size:
            weights = np.broadcast_to(weights, (self.size, self.size))
        shape = tuple(weights.shape)
        if shape != (self.size, self.size):
            error_msg = (
                f'The weights of the reduction {reduction.call} of observable {self.path} have the shape {shape}, but '
                f'the reduction needs the shape {(self.size, self.size)}, one row and one column per member of '
                f'{self.group}.'
            )
            raise ValidationError(error_msg)
        return scipy.sparse.csr_array(weights, dtype=float)

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
        entry = Entry(f'__observed_{index}', self.extent, shape, NumericDomain.REAL, members)
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
        >>> from skcomplex.engine.compile import ObservableEvaluator, compile_system
        >>> from skcomplex.core.system import Observable, System
        >>> from skcomplex.models.biology.ecology import PopulationComponent
        >>> system = System(components=[PopulationComponent.group('herd', 2, n=[2.0, 6.0])])
        >>> total = Observable('total', group='herd', reduce='sum', variable='n')
        >>> problem = compile_system(system, observables=[total])
        >>> ObservableEvaluator(problem).evaluate(0.0, problem.initial_state, problem.initial_parameters)
        {'total': 8.0}
    """

    problem: BaseProblem

    def __post_init__(self) -> None:
        """Compile every observable into a getter, in declaration order."""
        self.getters_: dict[str, Getter] = {}
        system = getattr(self.problem, 'system', None)
        observables = observables_of(system) if system is not None else ()
        if not isinstance(self.problem, Problem):
            getters = getattr(self.problem, 'observable_getters', None)
            self._scalars = {}
            self.getters_ = (getters(observables) if getters is not None else None) or {}
            return
        self._scalars = self._scalar_sources()
        own = self.problem.observable_getters(observables)
        if own is not None:
            self.getters_ = own
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
        stock_flow = self.problem.stock_flow
        for recorded in stock_flow.recorded if stock_flow is not None else ():
            self.getters_[recorded.path] = self._recorded_getter(recorded)

    @property
    def _compiled(self) -> Problem:
        """The problem as the compiler's problem, which the shared evaluator reads."""
        return cast('Problem', self.problem)

    def _scalar_sources(self) -> dict[Expression, tuple[str, int]]:
        """Return the vector and flat index of every symbol of a single element or interaction."""
        sources: dict[Expression, tuple[str, int]] = {}
        for name, layout in (('x', self.problem.states), ('p', self.problem.parameters)):
            for entry in layout:
                if entry.members is not None or entry.name not in self._compiled.symbols:
                    continue
                for column, item in enumerate(real_components_of(self._compiled.symbols[entry.name])):
                    sources[item] = (name, entry.start + column)
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

    def _recorded_getter(self, recorded: _Recorded) -> Getter:
        """Return the getter of a flow or an auxiliary, one value for a single component and one per group member."""
        members: dict[Expression, Callable[[Vector, Vector], Any]] = {}
        for item in symbols_of([recorded.expression]) - {self.problem.time}:
            reference = recorded.references.get(item)
            if reference is None or reference.source not in {Source.SINGLE, Source.MEMBER}:
                error_msg = (
                    f'{recorded.path} reads {item}, which is neither a variable of its component nor of a member of '
                    'its group, so trajectories cannot record it. Record it with an observable instead.'
                )
                raise ValidationError(error_msg)
            entry = reference.entry
            indices: Any = entry.start + reference.column
            if reference.entry.members is not None:
                indices = entry.start + np.arange(entry.members or 1) * entry.columns + reference.column
            members[item] = _from_vector(reference.is_state, indices)
        return self._compile(recorded.path, recorded.expression, {}, members)

    def _expression_getter(self, scope: str, path: str, observable: Observable) -> Getter:
        """Return the getter of an expression of the system's symbols."""
        prefix = f'{scope}/' if scope else ''
        items: dict[str, Any] = {
            name[len(prefix) :]: self._compiled.symbols[name]
            for name in self._compiled.symbols
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
        element = next((item for item in self._compiled.elements if item.path == group), None)
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
            for column, item in enumerate(real_components_of(self._compiled.symbols[name])):
                indices = (entry.start + np.arange(size) * entry.columns + column).astype(np.intp)
                members[item] = Read(is_state, indices, entry, column, entry.columns)
        namespace = Symbols(
            **{variable: self._compiled.symbols[name] for variable, name in names.items()},
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
        reductions = ReductionSchedule(tuple(lowering.found.values()), lowering.parameters)

        def read_expression(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            values = np.broadcast_to(evaluate(t, x, reductions.extend(t, x, p), done), (size,))
            return reduce_values(values, observable.reduce)

        return read_expression

    def _agent_getter(self, group: str, path: str, observable: Observable) -> Getter:
        """Return the getter of a reduction over an agent group, which the evaluation's agents give."""
        if not any(spec.path == group for spec in self._compiled.agents) and not self._compiled.partitions:
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


@dataclass
class Observer:
    """Values of the observables that laws read at the current state, built from the blocks of a system."""

    declarations: Declarations
    live: Declared
    values: dict[str, tuple[object, dict[Expression, Reference]]] = field(default_factory=dict)

    def _system_value(
        self,
        path: str,
        scope: str,
        observable: Observable,
    ) -> tuple[object, dict[Expression, Reference]]:
        """Return an expression of the symbols of a system's elements, with earlier observables by label."""
        prefix = f'{scope}/' if scope else ''
        items: dict[str, Any] = {}
        references: dict[Expression, Reference] = {}
        for block in self.declarations.blocks.values():
            for variable, entry in block.entries.items():
                if entry.members is None and entry.name.startswith(prefix):
                    items[entry.name[len(prefix) :]] = block.symbols[variable]
                    references.update(
                        {item: block.references[item] for item in real_components_of(block.symbols[variable])},
                    )
        namespace = _tree(items)
        for other, (other_scope, earlier) in self.live.items():
            if other == path:
                break
            if other_scope == scope:
                value, other_references = self.value(other, other_scope, earlier)
                setattr(namespace, earlier.label, value)
                references.update(other_references)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        return expression, references

    def value(self, path: str, scope: str, observable: Observable) -> tuple[object, dict[Expression, Reference]]:
        """Return the value of an observable with the references of the symbols it reads."""
        if path in self.values:
            return self.values[path]
        if observable.group is None:
            self.values[path] = self._system_value(path, scope, observable)
            return self.values[path]
        block = self.declarations.blocks[_join(scope, observable.group)]
        _, references = block.namespace(_source_of(block))
        self.values[path] = (_reduced_value(path, observable, block), references)
        return self.values[path]

    def __call__(self, node: Node, label: object) -> Observed | None:
        """Return the namespace of the observable a role binds to, or `None` for a role bound to no live observable."""
        path = f'{node.scope}{label}' if isinstance(label, str) else None
        if path not in self.live:
            return None
        scope, observable = self.live[path]
        value, references = self.value(path, scope, observable)
        return Observed(Symbols(value=value), dict(references))
