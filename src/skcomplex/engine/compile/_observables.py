"""Compile the observables of a system tree into one evaluator called at every frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Annotated, Any, Self

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import RR, SR, Expression, I
from sage.modules.free_module_element import FreeModuleElement

from ...core.elements import Element
from ...core.errors import RoleError, ValidationError
from ...core.laws.utils import Reduction, count, maximum, mean, minimum, reductions_in, total, variance
from ...core.specs import ObservableSpec, StateSpec
from ...core.system import System
from ...core.types import Vector
from ..problems import (
    BaseProblem,
    Entry,
    Form,
    Getter,
    Problem,
    ReductionSchedule,
    Reference,
    Source,
    VariableKind,
    declarations_of,
    numpy_function,
    symbol,
    symbols_of,
    vector,
)
from ..problems._quantities import _Recorded
from ..problems._reads import Read
from ..problems._reductions import CompiledReduction
from ._complex import _joined_complex
from ._declarations import Block, Declarations
from ._flows import _delay_calls
from ._specs import Observed
from ._tree import Node

type AgentsOf = Callable[[str], list[Any] | None]

_REDUCERS: dict[str, Callable[..., Expression]] = {
    'sum': total,
    'mean': mean,
    'max': maximum,
    'min': minimum,
    'var': variance,
}


type Declared = dict[str, tuple[str, ObservableSpec]]


def _join(scope: str, path: str) -> str:
    """Return a path relative to a system as a path from the root."""
    return f'{scope}/{path}' if scope else path


def _element_value(element: str, expression: Callable[[object], object], namespace: SimpleNamespace) -> object:
    """Return the value of an expression of an element's namespace and the time, read from its system's namespace."""
    return expression(SimpleNamespace(**vars(getattr(namespace, element)), t=namespace.t))


def _of_root(observable: ObservableSpec) -> ObservableSpec:
    """Return an observable of an element as the observable of the root system labelled `<element>.<label>`."""
    if observable.element is None or not callable(observable.expression):
        return observable
    expression = functools.partial(_element_value, observable.element, observable.expression)
    return dataclasses.replace(
        observable,
        label=f'{observable.element}.{observable.label}',
        expression=expression,
        element=None,
    )


def _with_observables(system: System, observables: Sequence[ObservableSpec]) -> System:
    """Return a copy of a system tree that declares observables in the systems their labels name."""
    by_scope: dict[str, list[ObservableSpec]] = {}
    for observable in observables:
        checked = _of_root(observable.check())
        scope, _, label = checked.label.rpartition('/')
        by_scope.setdefault(scope, []).append(dataclasses.replace(checked, label=label))
    if not by_scope:
        return system
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


def _attached(system: System, by_scope: dict[str, list[ObservableSpec]], scope: str) -> System:
    """Return a copy of a system that holds the observables of its scope, with its nested systems attached."""
    copied = copy.copy(system)
    copied.systems = [_attached(child, by_scope, _join(scope, child.label)) for child in system.systems]
    copied.observables = [*getattr(system, 'observables', ()), *by_scope.get(scope, ())]
    return copied


def _tree(items: Mapping[str, Any], **root: Any) -> SimpleNamespace:  # noqa: ANN401
    """Return nested namespaces from paths such as `forest/foxes.n` mapped to values, with extra values at the root."""
    nested: dict[str, Any] = {}
    for path, value in items.items():
        *systems, last = path.split('/')
        node = nested
        for name in systems:
            node = node.setdefault(name, {})
        element, _, variable = last.partition('.')
        node.setdefault(element, {})[variable] = value

    def build(node: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            **{key: build(value) if isinstance(value, dict) else value for key, value in node.items()},
        )

    return SimpleNamespace(**{key: build(value) for key, value in nested.items()}, **root)


def _entry_of(problem: BaseProblem, name: str) -> Entry | None:
    """Return the entry of a state or parameter of a problem by name, or `None`."""
    for layout in (problem.states, problem.parameters):
        if layout.has(name):
            return layout.entry(name)
    return None


def _is_lowerable(scope: str, observable: ObservableSpec, nodes: dict[str, Node]) -> bool:
    """Check whether laws can read an observable at the current state, as an expression or a symbolic reduction."""
    if observable.group is None:
        return True
    node = nodes.get(_join(scope, observable.group))
    return node is not None and not node.is_agent and isinstance(observable.reduce, str)


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


def _held_label(reduction: '_HeldReduction', labels: set[str]) -> str:
    """Return a label for the element that holds a reduction of a group, unique among the labels of its system."""
    base = f'{reduction.group.rpartition("/")[2]}_{reduction.reduce}'
    label, index = base, 1
    while label in labels:
        label, index = f'{base}_{index}', index + 1
    return label


def observables_of(system: System, scope: str = '') -> Iterator[tuple[str, ObservableSpec]]:
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


def namespace_of(problem: BaseProblem, scope: str = '') -> SimpleNamespace:
    """Return the namespace of the states and parameters of the single elements and interactions of a system.

    The namespace nests the symbols by their dotted names, so `s.plant.x` is the state `plant.x`, and
    `s.forest.foxes.n` the state `forest/foxes.n`. It also holds the time `s.t`.

    Args:
        problem:
            The compiled problem.

        scope:
            The path of the nested system whose symbols the namespace holds, empty for the root.

    Returns:
        namespace:
            The namespace.

    Examples:
        >>> from skcomplex.engine.compile import compile_system, namespace_of
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> namespace_of(compile_system(System(elements=[PopulationElement('rabbits')]))).rabbits.n
        n_rabbits
    """
    prefix = f'{scope}/' if scope else ''
    items: dict[str, Any] = {
        name[len(prefix) :]: problem.symbols[name]
        for name in problem.symbols
        if name.startswith(prefix) and (entry := _entry_of(problem, name)) is not None and entry.members is None
    }
    return _tree(items, t=problem.time)


def _lowerable_observables(system: System, nodes: list[Node]) -> Declared:
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


def _held_reduction(
    reduction: Reduction,
    agents: Mapping[Expression, tuple[str, str]],
    time: Expression,
    owner: str,
) -> '_HeldReduction | None':
    """Return the held reduction of a reduction of agents, or `None` for a reduction of elements.

    Args:
        reduction:
            The reduction a law reads.

        agents:
            The path of the agent group and the variable of every symbol of agents.

        time:
            The time symbol.

        owner:
            The path of the component whose law reads the reduction.

    Returns:
        held:
            The held reduction, or `None`.

    Raises:
        RoleError:
            The reduction reads the agents of more than one group.

        ValidationError:
            The reduction has weights, or reads a symbol that is not a variable of the agents.
    """
    expressions = [SR(reduction.argument), *([SR(reduction.condition)] if reduction.condition is not None else [])]
    symbols = symbols_of(expressions) - {time}
    read = symbols | ({reduction.over} if reduction.over is not None else set())
    groups = {agents[item][0] for item in read if item in agents}
    if not groups:
        return None
    if len(groups) > 1:
        error_msg = (
            f'The reduction {reduction.call} of {owner} reads the agents of {" and ".join(sorted(groups))}. A '
            'reduction reduces over one group, so write one reduction per group.'
        )
        raise RoleError(error_msg)
    group = next(iter(groups))
    others = sorted(str(item) for item in symbols if item not in agents or agents[item][0] != group)
    if reduction.weights is not None or others:
        error_msg = (
            f'The reduction {reduction.call} of {owner} reduces the agents of {group}, which a co-simulation computes '
            'at its synchronisation points from the scalar states and parameters of the agents alone, without '
            f'weights{", but it reads " + ", ".join(others) if others else ""}. Reduce an expression of the agents.'
        )
        raise ValidationError(error_msg)
    ordered = tuple(sorted((item for item in symbols if agents[item][1]), key=str))
    return _HeldReduction(
        group,
        reduction.kind,
        tuple(agents[item][1] for item in ordered),
        ordered,
        SR(reduction.argument),
        None if reduction.condition is None else SR(reduction.condition),
    )


def _with_held_reductions(system: System, reductions: Sequence['_HeldReduction'], scope: str = '') -> System:
    """Return a system tree that holds every reduction of agents in an element of the system of its group.

    Args:
        system:
            The system.

        reductions:
            The held reductions.

        scope:
            The path of the system, empty for the root.

    Returns:
        system:
            The system tree with the elements that hold the reductions.
    """
    own = [reduction for reduction in reductions if reduction.group.rpartition('/')[0] == scope]
    children = [_with_held_reductions(child, reductions, _join(scope, child.label)) for child in system.systems]
    if not own and all(new is old for new, old in zip(children, system.systems, strict=True)):
        return system
    labels = {part.label for part in (*system.elements, *system.interactions, *system.systems)}
    held = []
    for reduction in own:
        label = _held_label(reduction, labels)
        labels.add(label)
        held.append(_HeldElement(label, reduction=reduction))
    copied = copy.copy(system)
    copied.elements = [*system.elements, *held]
    copied.systems = children
    return copied


def _with_held_observables(system: System, read: set[str], scope: str = '') -> tuple[System, list[str]]:
    """Return a system tree in which every observable in `read` is also held in an element, and their paths."""
    held: list[str] = []
    elements = []
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
        variables = (observable.variable,) if observable.variable is not None else ()
        reduction = _HeldReduction(_join(scope, observable.group), observable.reduce, variables)
        elements.append(_HeldElement(observable.label, reduction=reduction))
        held.append(path)
    children = []
    for child in system.systems:
        new_child, child_held = _with_held_observables(child, read, f'{scope}/{child.label}' if scope else child.label)
        children.append(new_child)
        held += child_held
    if not held:
        return system, held
    copied = copy.copy(system)
    copied.elements = [*system.elements, *elements]
    copied.systems = children
    return copied, held


def _held_observables(system: System, nodes: list[Node]) -> set[str]:
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
    lowerable = _lowerable_observables(system, nodes)
    declared = {_join(scope, observable.label) for scope, observable in observables_of(system)}
    return (_role_paths(system) & declared) - set(lowerable)


def held_reductions_of(problem: BaseProblem) -> list[tuple[str, '_HeldReduction']]:
    """Return the reductions that a co-simulation computes at every synchronisation point, with where it holds them.

    A law of one partition reads a reduction of the agents of another, such as `count(s.farmers)`, through an element
    that holds its value in the state `value` between synchronisation points.

    Args:
        problem:
            The problem.

    Returns:
        reductions:
            The path of every element that holds a reduction, with the reduction.

    Examples:
        >>> from skcomplex.engine.compile import compile_system, held_reductions_of
        >>> from skcomplex.models import create
        >>> [path for path, _ in held_reductions_of(compile_system(create('harvesting')))]
        ['village/farmers_count']
    """
    return [
        (element.path, element.element.reduction)
        for element in problem.elements
        if isinstance(element.element, _HeldElement)
    ]


def _check_held_reads(problem: Problem, held: list[str]) -> None:
    """Raise an error when a law reads an observable that cannot be lowered from the partition that computes it."""
    groups = {
        _join(scope, observable.label): _join(scope, observable.group or '')
        for scope, observable in observables_of(problem.system)  # type: ignore[arg-type]
    }
    for path in held:
        readers = _readers_of(problem, path)
        shared = [item.name for item in problem.partitions if groups[path] in item.owners and readers & item.owners]
        if problem.form != Form.COUPLED or shared:
            error_msg = (
                f'A law reads the observable {path} in the partition that computes it, but the observable is a Python '
                'function of the values or reduces agents, which laws cannot read at the current state. Write it '
                'symbolically with a variable or an expression of the members, or move the reading law into another '
                'subsystem of a different form.'
            )
            raise ValidationError(error_msg)


class _HeldElement(Element):
    """Element that holds the value of a held reduction in its state `value`, which laws of other partitions read."""

    def __init__(
        self,
        label: str,
        value: Annotated[float, StateSpec()] = 0.0,
        *,
        reduction: '_HeldReduction | None' = None,
    ) -> None:
        super().__init__(label)
        self.value = value
        self.reduction = reduction


@dataclass(frozen=True)
class _HeldReduction:
    """Reduction of a group that a co-simulation computes at every synchronisation point and holds until the next."""

    group: str
    reduce: str | Callable[[Any], Any]
    variables: tuple[str, ...] = ()
    symbols: tuple[Expression, ...] = ()
    argument: Expression | None = None
    condition: Expression | None = None

    @property
    def key(self) -> tuple[str, ...]:
        """Group, reduction, argument, and condition, which identify the reduction."""
        return (self.group, str(self.reduce), str(self.argument), str(self.condition))

    def reducer(self, time: Expression) -> Callable[[float, Sequence[Vector], Sequence[Any]], float]:
        """Return the function of time, the values of the variables, and the members that computes the reduction.

        Args:
            time:
                The time symbol.

        Returns:
            reducer:
                The function.
        """
        reduce = self.reduce
        if self.argument is None:

            def reduce_values_of(t: float, columns: Sequence[Vector], members: Sequence[Any]) -> float:
                del t
                return float(np.real(CompiledReduction._reduce_values(columns[0] if columns else members, reduce)))

            return reduce_values_of
        condition = SR(1) if self.condition is None else self.condition
        function = numpy_function(time, list(self.symbols), [self.argument, condition])

        def reduce_expression(t: float, columns: Sequence[Vector], members: Sequence[Any]) -> float:
            size = len(members)
            argument, selected = (
                np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in function(t, *columns)
            )
            return float(np.real(CompiledReduction._reduce_values(argument[selected.astype(bool)], reduce)))

        return reduce_expression


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
        entry = Entry(f'__observed_{index}', self.extent, shape, RR, members)
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


class ObservableEvaluator:
    """Evaluator of every observable of a problem from a state vector, a parameter vector, and the agents.

    Preparing compiles every observable into a getter, in declaration order, the prepared `getters_`.

    Args:
        problem:
            The compiled problem.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.engine.compile import ObservableEvaluator, compile_system
        >>> from skcomplex.core.specs import ObservableSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> system = System(
        ...     elements=[GroupElement('herd', element=PopulationElement, size=2, values={'n': [2.0, 6.0]})]
        ... )
        >>> total = ObservableSpec('total', group='herd', reduce='sum', variable='n')
        >>> problem = compile_system(system, observables=[total])
        >>> ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
        {'total': 8.0}
    """

    def __init__(self, problem: BaseProblem) -> None:
        self.problem = problem

    def prepare(self) -> Self:
        """Compile every observable into a getter, in declaration order.

        Returns:
            evaluator:
                The evaluator, with the getter of every observable path in `getters_`.

        Raises:
            ValidationError:
                Two observables share a path, an observable has neither an expression nor a group, or an expression
                reads a symbol it cannot.
        """
        self.getters_: dict[str, Getter] = {}
        system = self.problem.system
        observables = observables_of(system) if system is not None else ()
        own = self.problem.observable_getters(observables)
        if own is not None:
            self._scalars = {}
            self.getters_ = own
            return self
        self._scalars = self._scalar_sources()
        for scope, observable in observables_of(system) if system is not None else ():
            path = _join(scope, observable.label)
            if path in self.getters_:
                error_msg = f'Two observables have the path {path}. Give observables of one system distinct labels.'
                raise ValidationError(error_msg)
            if observable.group is None and observable.expression is None:
                error_msg = f'ObservableSpec {path} has neither an expression nor a group. Give one of them.'
                raise ValidationError(error_msg)
            if observable.group is None:
                self.getters_[path] = self._expression_getter(scope, path, observable)
            else:
                self.getters_[path] = self._reduction_getter(scope, path, observable)
        problems = (self.problem, *(partition.problem for partition in self.problem.partitions))
        for renewal in (renewal for problem in problems for renewal in problem.renewals):
            self.getters_[f'{renewal.label}.renewals'] = self._renewal_counter(renewal.counts)
        stock_flow = self.problem.stock_flow
        for recorded in stock_flow.recorded if stock_flow is not None else ():
            self.getters_[recorded.path] = self._recorded_getter(recorded)
        return self

    def _scalar_sources(self) -> dict[Expression, tuple[str, int]]:
        """Return the vector and flat index of every symbol of a single element or interaction."""
        sources: dict[Expression, tuple[str, int]] = {}
        for name, layout in (('x', self.problem.states), ('p', self.problem.parameters)):
            for entry in layout.entries:
                if entry.members is not None or entry.name not in self.problem.symbols:
                    continue
                for column, item in enumerate(Problem._real_components_of(self.problem.symbols[entry.name])):
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
        expression = _joined_complex(expression, f'observable {path}')
        arguments = sorted(symbols_of([expression]) - {self.problem.time}, key=str)
        readers: list[Callable[[Vector, Vector, Mapping[str, Any]], Any]] = []
        for item in arguments:
            if item in observables:
                readers.append(self._from_done(observables[item]))
            elif item in members:
                readers.append(self._from_members(members[item]))
            elif item in self._scalars:
                vector, index = self._scalars[item]
                readers.append(self._from_members(self._from_vector(vector == 'x', index)))
            else:
                error_msg = (
                    f'ObservableSpec {path} uses the symbol {item}, which is neither a variable of a single element or '
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
        """Return the getter of a flow or an objective, one value for a single component and one per group member."""
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
            members[item] = self._from_vector(reference.is_state, indices)
        return self._compile(recorded.path, recorded.expression, {}, members)

    def _expression_getter(self, scope: str, path: str, observable: ObservableSpec) -> Getter:
        """Return the getter of an expression of the system's symbols."""
        prefix = f'{scope}/' if scope else ''
        earlier = {
            other: symbol(f'_observable_{index}')
            for index, other in enumerate(self.getters_)
            if other.startswith(prefix) and '/' not in other[len(prefix) :]
        }
        namespace = namespace_of(self.problem, scope)
        for other, item in earlier.items():
            setattr(namespace, other[len(prefix) :], item)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        expression, stages = self._read_delays(path, expression)
        return self._compile(path, expression, {item: other for other, item in earlier.items()}, stages)

    def _read_delays(
        self,
        path: str,
        expression: Any,  # noqa: ANN401
    ) -> tuple[Any, dict[Expression, Callable[[Vector, Vector], Any]]]:
        """Return an expression with the delays of the laws read from their hidden stocks, and the stocks' readers."""
        stock_flow = self.problem.stock_flow
        if stock_flow is None or not isinstance(expression, Expression):
            return expression, {}
        references: dict[Expression, Reference] = {}
        for delay in stock_flow.delays:
            expression = SR(expression).subs({delay.call: delay.output})
            references.update(delay.references)
        if _delay_calls(expression):
            error_msg = (
                f'ObservableSpec {path} reads {_delay_calls(expression)[0]}, a delay that no law of the system uses. '
                'Observe a delay that a law of the same element reads.'
            )
            raise ValidationError(error_msg)
        stages = {
            item: self._from_vector(True, references[item].entry.start + references[item].column)
            for item in symbols_of([expression])
            if item in references
        }
        return expression, stages

    def _entry(self, name: str) -> Any:  # noqa: ANN401
        """Return the entry of a state or parameter by name, or `None`."""
        return _entry_of(self.problem, name)

    def _reduction_getter(self, scope: str, path: str, observable: ObservableSpec) -> Getter:
        """Return the getter of a reduction over a group of elements or agents."""
        group = _join(scope, observable.group or '')
        element = next((item for item in self.problem.elements if item.path == group), None)
        if element is None:
            return self._agent_getter(group, path, observable)
        names = {**element.states, **element.parameters}
        if observable.variable is not None:
            if observable.variable not in names:
                error_msg = f'ObservableSpec {path} reduces {observable.variable}, which {group} does not declare.'
                raise ValidationError(error_msg)
            name = names[observable.variable]
            is_state = self.problem.states.has(name)
            layout = self.problem.states if is_state else self.problem.parameters

            def read_variable(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
                del t, done
                values = np.asarray(layout.read(x if is_state else p, name))
                return CompiledReduction._reduce_values(
                    values if element.members is not None else values[np.newaxis], observable.reduce,
                )

            return read_variable
        size = element.members or 1
        if observable.expression is None:
            if observable.reduce != 'count':
                error_msg = f'ObservableSpec {path} reduces {group} with {observable.reduce!r} but names no variable.'
                raise ValidationError(error_msg)
            return lambda t, x, p, done: float(size)
        members: dict[Expression, Read] = {}
        for name in names.values():
            entry = self._entry(name)
            is_state = self.problem.states.has(name)
            for column, item in enumerate(Problem._real_components_of(self.problem.symbols[name])):
                indices = (entry.start + np.arange(size) * entry.columns + column).astype(np.intp)
                members[item] = Read(is_state, indices, entry, column, entry.columns)
        namespace = SimpleNamespace(
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
        readers = {item: self._read_of(read) for item, read in lowering.reads.items() if item not in scalars}
        evaluate = self._compile(path, expression, {}, readers)
        reductions = ReductionSchedule(tuple(lowering.found.values()), lowering.parameters)

        def read_expression(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            values = np.broadcast_to(evaluate(t, x, reductions.extend(t, x, p), done), (size,))
            return CompiledReduction._reduce_values(values, observable.reduce)

        return read_expression

    def _agent_getter(self, group: str, path: str, observable: ObservableSpec) -> Getter:
        """Return the getter of a reduction over an agent group, which the evaluation's agents give."""
        if not any(spec.path == group for spec in self.problem.agents) and not self.problem.partitions:
            error_msg = f'ObservableSpec {path} reduces {group}, which is neither an element group nor an agent group.'
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
            return CompiledReduction._reduce_values(values, observable.reduce)

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
                A function that returns the views of the agents of an agent group by path, or `None` without agents.

        Returns:
            values:
                A mapping from observable path to its value.
        """
        done: dict[str, Any] = {'__agents__': agents_of or (lambda group: None)}
        for path, get in self.getters_.items():
            done[path] = get(t, x, p, done)
        del done['__agents__']
        return done

    @staticmethod
    def _from_done(name: str) -> Callable[[Vector, Vector, Mapping[str, Any]], Any]:
        """Return a reader of an observable computed earlier in the same frame."""
        return lambda x, p, done: done[name]

    @staticmethod
    def _from_members(read: Callable[[Vector, Vector], Any]) -> Callable[[Vector, Vector, Mapping[str, Any]], Any]:
        """Return a reader of one member variable over a group."""
        return lambda x, p, done: read(x, p)

    @staticmethod
    def _read_of(read: Read) -> Callable[[Vector, Vector], Any]:
        """Return a reader of the values a reduction read stands for, from the state or the extended parameters."""
        return ObservableEvaluator._from_vector(read.is_state, read.index)

    @staticmethod
    def _from_vector(is_state: bool, indices: Any) -> Callable[[Vector, Vector], Any]:  # noqa: ANN401
        """Return a reader of components of the state or the parameter vector."""
        return lambda x, p: (x if is_state else p)[indices]

    @staticmethod
    def _renewal_counter(counts: npt.NDArray[np.intp]) -> Getter:
        """Return the getter of the number of renewals of a renewal law over all its members."""

        def count_renewals(t: float, x: Vector, p: Vector, done: Mapping[str, Any]) -> float:
            del t, p, done
            return float(np.sum(x[counts]))

        return count_renewals


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
        observable: ObservableSpec,
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
                        {item: block.references[item] for item in Problem._real_components_of(block.symbols[variable])},
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

    def value(
        self,
        path: str,
        scope: str,
        observable: ObservableSpec,
    ) -> tuple[object, dict[Expression, Reference]]:
        """Return the value of an observable with the references of the symbols it reads."""
        if path in self.values:
            return self.values[path]
        if observable.group is None:
            self.values[path] = self._system_value(path, scope, observable)
            return self.values[path]
        block = self.declarations.blocks[_join(scope, observable.group)]
        _, references = block.namespace(self._source_of(block))
        self.values[path] = (self._reduced_value(path, observable, block), references)
        return self.values[path]

    def evaluate(self, node: Node, label: object) -> Observed | None:
        """Return the namespace of the observable a role binds to, or `None` for a role bound to no live observable."""
        path = f'{node.scope}{label}' if isinstance(label, str) else None
        if path not in self.live:
            return None
        scope, observable = self.live[path]
        value, references = self.value(path, scope, observable)
        return Observed(SimpleNamespace(value=value), dict(references))

    @staticmethod
    def _source_of(block: Block) -> Source:
        """Return how laws read the variables of a block from outside it, per member for a group."""
        return Source.MEMBER if block.node.members is not None else Source.SINGLE

    @staticmethod
    def _reduced_value(path: str, observable: ObservableSpec, block: Block) -> object:
        """Return an observable that reduces a group as a reduction over the group's member symbols."""
        namespace, _ = block.namespace(Observer._source_of(block))
        if observable.reduce == 'count':
            return count(over=namespace)
        if observable.variable is not None:
            if observable.variable not in block.symbols:
                error_msg = (
                    f'ObservableSpec {path} reduces {observable.variable}, which {block.node.path} does not declare.'
                )
                raise ValidationError(error_msg)
            value: Any = block.symbols[observable.variable]
        elif observable.expression is None:
            error_msg = (
                f'ObservableSpec {path} reduces {block.node.path} with {observable.reduce!r} but names no variable.'
            )
            raise ValidationError(error_msg)
        else:
            value = observable.expression(namespace) if callable(observable.expression) else observable.expression
        reducer = _REDUCERS[str(observable.reduce)]
        if isinstance(value, FreeModuleElement):
            return vector(reducer(item, over=namespace) for item in value)
        return reducer(value, over=namespace)
