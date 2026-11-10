"""Compile the observables of a system tree into one evaluator.

The evaluator is called at every frame.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, I

from ...components.elements import Element
from ...components.laws.utils import local_operator
from ...components.laws.utils._local_operators import has_operator
from ...components.specs import NodeSpec
from ...components.systems import CompositeSystem
from ...errors import ValidationError
from ...specs import EntrySpec, NeighborReadSpec, ObservableSpec, ReadSpec, RunStateSpec, StateSpec
from ...specs._held_reduction_spec import _HeldReductionSpec
from ...types import Array, FormKind, Indices, VariableKind
from .._utils import bound_labels, declarations_of, symbol, symbols_of
from ..problems import BaseProblemSpec, ProblemSpec
from ..problems._drive_kernel import DriveKernel
from ..problems._expression_printer import ExpressionPrinter
from ..problems._law_kernel import LawKernel
from ..problems._operator_kernel import OperatorKernel
from ..problems._operator_lowering import OperatorLowering
from ..problems._operator_problem_spec import OperatorProblemSpec
from ..problems._recorded_spec import _RecordedSpec
from ..problems._reduction_kernel import ReductionKernel
from ..problems._reduction_schedule_spec import ReductionScheduleSpec
from ..problems._reference_spec import ReferenceSpec
from ..types import AgentsFunction, Declared, ObservableFunction, ReadKind
from ._base import BaseCompiler, BasePass
from ._flows import TransferLowering
from ._member_reductions import _MemberReductions
from ._members import MemberLowering
from .passes._complex import LowerComplexPass


class _HeldElement(Element):
    """Element that holds the value of a held reduction in its state `value`.

    Laws of other partitions read this state.
    """

    def __init__(
        self,
        label: str,
        value: Annotated[float, StateSpec()] = 0.0,
        *,
        reduction: '_HeldReductionSpec | None' = None,
    ) -> None:
        super().__init__(label)
        self.value = value
        self.reduction = reduction


class ObservableEvaluator:
    """Evaluator of every observable of a problem.

    It evaluates them from a state vector, a parameter vector, and the agents. Preparing compiles every observable into
    a getter, in declaration order, the prepared `getters_`.

    Args:
        problem:
            The compiled problem.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.engine.compile._observables import ObservableEvaluator
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.core.specs import ObservableSpec
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> system = CompositeSystem(
        ...     elements=[GroupElement('herd', element=PopulationElement, size=2, values={'x': [2.0, 6.0]})]
        ... )
        >>> total = ObservableSpec('total', group='herd', reduce='sum', variable='x')
        >>> problem = PassCompiler(observables=[total]).prepare(system).run().problem_
        >>> ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
        {'total': 8.0}
    """

    @staticmethod
    def _get_role_paths(system: CompositeSystem, scope: str = '') -> set[str]:
        """Return the paths from the root of everything roles of a system tree bind to."""
        paths = set()
        for interaction in system.interactions:
            for name in declarations_of(interaction, VariableKind.ROLE):
                labels = bound_labels(getattr(interaction, name))
                paths |= {f'{scope}/{label}' if scope else label for label in labels}
        for child in system.systems:
            paths |= ObservableEvaluator._get_role_paths(child, f'{scope}/{child.label}' if scope else child.label)
        return paths

    @staticmethod
    def _get_readers_of(problem: ProblemSpec, path: str) -> set[str]:
        """Return the paths that interactions reading an observable bind other roles to.

        The paths are those of elements and groups.
        """
        readers: set[str] = set()
        for interaction in problem.interactions:
            bound = [
                item for value in interaction.roles.values() for item in ([value] if isinstance(value, str) else value)
            ]
            if path in bound:
                readers |= set(bound) - {path}
        return readers

    @staticmethod
    def _get_held_label(reduction: '_HeldReductionSpec', labels: set[str]) -> str:
        """Return a label for the element that holds a reduction of a group.

        The label is unique among the labels of its system.
        """
        base = f'{reduction.group.rpartition("/")[2]}_{reduction.reduce}'
        label, index = base, 1
        while label in labels:
            label, index = f'{base}_{index}', index + 1
        return label

    @staticmethod
    def _copy_with_held_reductions(
        system: CompositeSystem,
        reductions: Sequence['_HeldReductionSpec'],
        scope: str = '',
    ) -> CompositeSystem:
        """Return a system tree that holds every reduction of agents in an element.

        The element is in the system of the group.

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
        children = [
            ObservableEvaluator._copy_with_held_reductions(
                child,
                reductions,
                ObservableEvaluator._join(scope, child.label),
            )
            for child in system.systems
        ]
        if not own and all(new is old for new, old in zip(children, system.systems, strict=True)):
            return system
        labels = {part.label for part in (*system.elements, *system.interactions, *system.systems)}
        held = []
        for reduction in own:
            label = ObservableEvaluator._get_held_label(reduction, labels)
            labels.add(label)
            held.append(_HeldElement(label, reduction=reduction))
        copied = copy.copy(system)
        copied.elements = [*system.elements, *held]
        copied.systems = children
        return copied

    @staticmethod
    def _copy_with_held_observables(
        system: CompositeSystem,
        read: set[str],
        scope: str = '',
    ) -> tuple[CompositeSystem, list[str]]:
        """Return a system tree that also holds every observable in `read` in an element.

        It also returns their paths.
        """
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
            reduction = _HeldReductionSpec(
                ObservableEvaluator._join(scope, observable.group),
                observable.reduce,
                variables,
            )
            elements.append(_HeldElement(observable.label, reduction=reduction))
            held.append(path)
        children = []
        for child in system.systems:
            new_child, child_held = ObservableEvaluator._copy_with_held_observables(
                child,
                read,
                f'{scope}/{child.label}' if scope else child.label,
            )
            children.append(new_child)
            held += child_held
        if not held:
            return system, held
        copied = copy.copy(system)
        copied.elements = [*system.elements, *elements]
        copied.systems = children
        return copied, held

    @staticmethod
    def _get_held_observables(system: CompositeSystem, nodes: list[NodeSpec]) -> set[str]:
        """Return the paths of the observables read only at synchronisation points.

        They are the observables that roles bind to and laws read only at those points.

        Args:
            system:
                The system, with the observables of the run.

            nodes:
                The nodes of the system.

        Returns:
            paths:
                The paths of the held observables.
        """
        lowerable = ObservableEvaluator._get_lowerable_observables(system, nodes)
        declared = {
            ObservableEvaluator._join(scope, observable.label)
            for scope, observable in BaseCompiler._get_observables_of(system)
        }
        return (ObservableEvaluator._get_role_paths(system) & declared) - set(lowerable)

    @staticmethod
    def _check_held_reads(problem: ProblemSpec, held: list[str]) -> None:
        """Raise an error for a law that reads an observable that cannot be lowered.

        The observable cannot be lowered from the partition that computes it.
        """
        if problem.system is None:
            return
        groups = {
            ObservableEvaluator._join(scope, observable.label): ObservableEvaluator._join(scope, observable.group or '')
            for scope, observable in BaseCompiler._get_observables_of(problem.system)
        }
        for path in held:
            readers = ObservableEvaluator._get_readers_of(problem, path)
            partitions = problem.partitions if isinstance(problem, ProblemSpec) else ()
            shared = [item.name for item in partitions if groups[path] in item.owners and readers & item.owners]
            if problem.form != FormKind.COUPLED or shared:
                error_msg = (
                    f'A law reads the observable {path} in the partition that computes it, but the observable is a '
                    'Python '
                    'function of the values or reduces agents, which laws cannot read at the current state. Write it '
                    'symbolically with a variable or an expression of the members, or move the reading law into '
                    'another '
                    'subsystem of a different form.'
                )
                raise ValidationError(error_msg)

    def __init__(self, problem: BaseProblemSpec) -> None:
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
        self.getters_: dict[str, ObservableFunction] = {}
        if isinstance(self.problem, OperatorProblemSpec):
            self._scalars = {}
            self.getters_ = self._get_operator_getters(self.problem)
            return self
        self._scalars = self._get_scalar_sources()
        for scope, observable in self.problem.observables:
            path = self._join(scope, observable.label)
            if path in self.getters_:
                error_msg = f'Two observables have the path {path}. Give observables of one system distinct labels.'
                raise ValidationError(error_msg)
            if observable.function is not None:
                self.getters_[path] = self._get_function_getter(scope, path, observable.function)
                continue
            if observable.group is None and observable.expression is None:
                error_msg = (
                    f'ObservableSpec {path} has neither an expression, a function, nor a group. Give one of them.'
                )
                raise ValidationError(error_msg)
            observed = self.problem.observed if isinstance(self.problem, ProblemSpec) else {}
            if observable.group is None:
                self.getters_[path] = self._get_expression_getter(scope, path, observable)
            elif path in observed:
                self.getters_[path] = self._get_observed_getter(observed[path], observable)
            else:
                self.getters_[path] = self._get_reduction_getter(scope, path, observable)
        self._add_law_records()
        return self

    def _add_law_records(self) -> None:
        """Add the getters of what laws record.

        They record the renewals, the flows and objectives, and the scores of drives.
        """
        problems = (self.problem, *self.problem.partition_problems)
        for renewal in (renewal for problem in problems for renewal in problem.renewals):
            self.getters_[f'{renewal.label}.renewals'] = self._get_renewal_counter(renewal.counts)
        transfer_network = self.problem.transfer_network
        for recorded in transfer_network.recorded if transfer_network is not None else ():
            self.getters_[recorded.path] = self._get_recorded_getter(recorded)
        for problem in problems:
            for step in problem.lattice if isinstance(problem, ProblemSpec) else ():
                if isinstance(step, DriveKernel) and step.score is not None:
                    self.getters_[f'{step.owner}.score'] = self._score_getter(step)

    def _score_getter(self, drive: DriveKernel) -> ObservableFunction:
        """Return the getter of the scores of the rows of a lattice state."""

        def get(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del t, done
            return drive.read_score(x, p)

        return get

    def _get_function_getter(
        self,
        scope: str,
        path: str,
        function: Callable[[float, Mapping[str, object]], object],
    ) -> ObservableFunction:
        """Return a getter that calls a function of time and the system values.

        The values are those of the states and parameters of its system.
        """
        problem = self.problem
        if not isinstance(problem, ProblemSpec) or problem.agents:
            error_msg = (
                f'ObservableSpec {path} is a function of the values of a system, which a system with agents or a '
                'operator system does not hold as a vector. Give an expression, or an expression of each agent.'
            )
            raise ValidationError(error_msg)
        states, parameters = problem.states, problem.parameters
        prefix = f'{scope}/' if scope else ''

        def get(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del done
            values = {**states.get_values(x), **parameters.get_values(p)}
            local = {name.removeprefix(prefix): item for name, item in values.items() if name.startswith(prefix)}
            value = np.asarray(function(t, local))
            return float(value) if value.ndim == 0 else value

        return get

    def _get_scalar_sources(self) -> dict[Expression, tuple[str, int]]:
        """Return the vector and flat index of every single element or interaction symbol."""
        sources: dict[Expression, tuple[str, int]] = {}
        for name, layout in (('x', self.problem.states), ('p', self.problem.parameters)):
            for entry in layout.entries:
                if entry.members is not None or entry.name not in self.problem.symbols:
                    continue
                for column, item in enumerate(ProblemSpec._get_real_components_of(self.problem.symbols[entry.name])):
                    sources[item] = (name, entry.start + column)
        return sources

    def _compile(
        self,
        path: str,
        expression: object,
        observables: Mapping[Expression, str],
        members: Mapping[Expression, Callable[[Array, Array], Array]],
    ) -> ObservableFunction:
        """Return a getter that evaluates an expression.

        The expression is evaluated from scalars, observables, and member arrays.
        """
        expression = self._get_joined_complex(expression, f'observable {path}')
        arguments = sorted(symbols_of([expression]) - {self.problem.time}, key=str)
        readers: list[Callable[[Array, Array, Mapping[str, object]], object]] = []
        for item in arguments:
            if item in observables:
                readers.append(self._create_from_done(observables[item]))
            elif item in members:
                readers.append(self._create_from_members(members[item]))
            elif item in self._scalars:
                vector, index = self._scalars[item]
                readers.append(self._create_from_members(self._create_from_vector(vector == 'x', index)))
            else:
                error_msg = (
                    f'ObservableSpec {path} uses the symbol {item}, which is neither a variable of a single element or '
                    'interaction nor an earlier observable. Use the namespace that the expression receives.'
                )
                raise ValidationError(error_msg)
        function = ExpressionPrinter(time=self.problem.time, compiled=True).prepare(arguments).run([expression]).code_
        if not callable(function):
            error_msg = f'A compiled printer returns a function, not a {type(function).__name__}.'
            raise TypeError(error_msg)

        def get(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            values = function(t, *(read(x, p, done) for read in readers))
            if not isinstance(values, list):
                error_msg = f'A compiled function returns a list of values, not a {type(values).__name__}.'
                raise TypeError(error_msg)
            value = np.real_if_close(values[0])
            return float(value) if np.ndim(value) == 0 and np.isrealobj(value) else value

        return get

    def _get_recorded_getter(self, recorded: _RecordedSpec) -> ObservableFunction:
        """Return the getter of a flow or an objective.

        It gives one value for a single component and one per group member. A flow along the edges of a group reads
        members at an offset, which the getter reads from the member columns.
        """
        expression, members = self._read_offsets(recorded)
        for item in symbols_of([expression]) - {self.problem.time} - set(members):
            reference = recorded.references.get(item)
            if reference is None or reference.source not in {ReadKind.SINGLE, ReadKind.MEMBER}:
                error_msg = (
                    f'{recorded.path} reads {item}, which is neither a variable of its component nor of a member of '
                    'its group, so trajectories cannot record it. Record it with an observable instead.'
                )
                raise ValidationError(error_msg)
            entry = reference.entry
            indices: int | Indices = entry.start + reference.column
            if reference.entry.members is not None and reference.source is ReadKind.MEMBER:
                indices = entry.start + np.arange(entry.members or 1) * entry.columns + reference.column
            members[item] = self._create_from_vector(reference.is_state, indices)
        return self._compile(recorded.path, expression, {}, members)

    def _read_offsets(
        self,
        recorded: _RecordedSpec,
    ) -> tuple[object, dict[Expression, Callable[[Array, Array], Array]]]:
        """Return a recorded expression with member offset reads replaced by symbols.

        It also returns its reader.
        """
        expression = recorded.expression
        if not isinstance(expression, Expression):
            return expression, {}
        members: dict[Expression, Callable[[Array, Array], Array]] = {}
        for index, call in enumerate(BasePass._get_calls_of(expression, 'shift_member')):
            base, offset, boundary, fill = MemberLowering._shift_arguments(call)
            reference = recorded.references.get(base)
            if reference is None or reference.source is not ReadKind.MEMBER:
                error_msg = f'{recorded.path} reads {base} at a member offset, but it is no variable of a group member.'
                raise ValidationError(error_msg)
            read = MemberLowering._get_neighbor_read(reference.entry.rows, offset, boundary, fill)
            column = self._create_from_vector(reference.is_state, reference.flat_indices)
            item = symbol(f'_recorded_shift{index}')
            members[item] = self._create_from_offset(column, read)
            expression = SR(expression).subs({call: item})
        return expression, members

    def _get_expression_getter(self, scope: str, path: str, observable: ObservableSpec) -> ObservableFunction:
        """Return the getter of an expression of the system's symbols."""
        prefix = f'{scope}/' if scope else ''
        earlier = {
            other: symbol(f'_observable_{index}')
            for index, other in enumerate(self.getters_)
            if other.startswith(prefix) and '/' not in other[len(prefix) :]
        }
        namespace = self._get_namespace_of(self.problem, scope)
        for other, item in earlier.items():
            setattr(namespace, other[len(prefix) :], item)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        expression, stages = self._read_delays(path, expression)
        return self._compile(path, expression, {item: other for other, item in earlier.items()}, stages)

    def _read_delays(
        self,
        path: str,
        expression: object,
    ) -> tuple[object, dict[Expression, Callable[[Array, Array], Array]]]:
        """Return an expression with the delays read from their hidden states.

        The delays are those of the laws, and their readers are also returned.
        """
        transfer_network = self.problem.transfer_network
        if transfer_network is None or not isinstance(expression, Expression):
            return expression, {}
        references: dict[Expression, ReferenceSpec] = {}
        for delay in transfer_network.delays:
            expression = SR(expression).subs({delay.call: delay.output})
            references.update(delay.references)
        delays = TransferLowering._get_delay_calls(expression)
        if delays:
            error_msg = (
                f'ObservableSpec {path} reads {delays[0]}, a delay that no law of the '
                'system uses. Observe a delay that a law of the same element reads.'
            )
            raise ValidationError(error_msg)
        stages = {
            item: self._create_from_vector(True, references[item].entry.start + references[item].column)
            for item in symbols_of([expression])
            if item in references
        }
        return expression, stages

    def _get_entry(self, name: str) -> EntrySpec:
        """Return the entry of a state or parameter by name.

        Args:
            name:
                The dotted name of the state or parameter.

        Returns:
            entry:
                The entry.

        Raises:
            ValidationError:
                The problem has no state or parameter of that name.
        """
        entry = self._get_entry_of(self.problem, name)
        if entry is None:
            error_msg = f'The problem has no state or parameter named {name!r}.'
            raise ValidationError(error_msg)
        return entry

    def _get_reduction_getter(self, scope: str, path: str, observable: ObservableSpec) -> ObservableFunction:
        """Return the getter of a reduction over a group of elements or agents."""
        group = self._join(scope, observable.group or '')
        element = next((item for item in self.problem.elements if item.path == group), None)
        if element is None:
            return self._get_agent_getter(group, path, observable)
        names = {**element.states, **element.parameters}
        if observable.variable is not None:
            if observable.variable not in names:
                error_msg = f'ObservableSpec {path} reduces {observable.variable}, which {group} does not declare.'
                raise ValidationError(error_msg)
            name = names[observable.variable]
            is_state = self.problem.states.has(name)
            layout = self.problem.states if is_state else self.problem.parameters

            def read_variable(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
                del t, done
                values = np.asarray(layout.read(x if is_state else p, name))
                return ReductionKernel._reduce_values(
                    values if element.members is not None else values[np.newaxis],
                    observable.reduce,
                )

            return read_variable
        size = element.members or 1
        if observable.expression is None:
            if observable.reduce != 'count':
                error_msg = f'ObservableSpec {path} reduces {group} with {observable.reduce!r} but names no variable.'
                raise ValidationError(error_msg)
            return lambda t, x, p, done: float(size)
        members: dict[Expression, ReadSpec] = {}
        for name in names.values():
            entry = self._get_entry(name)
            is_state = self.problem.states.has(name)
            for column, item in enumerate(ProblemSpec._get_real_components_of(self.problem.symbols[name])):
                indices = (entry.start + np.arange(size) * entry.columns + column).astype(np.intp)
                members[item] = ReadSpec(is_state, indices, entry, column, entry.columns)
        namespace = SimpleNamespace(
            **{variable: self.problem.symbols[name] for variable, name in names.items()},
            t=self.problem.time,
        )
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        scalars = {item: ReadSpec(vector == 'x', index) for item, (vector, index) in self._scalars.items()}
        parameters = len(self.problem.initial_parameters)
        lowering = _MemberReductions(
            path=path,
            group=group,
            size=size,
            reads={**scalars, **members},
            time=self.problem.time,
            parameters=parameters,
        )
        expression = lowering.prepare().run(expression).expression_
        readers = {item: self._read_of(read) for item, read in lowering.reads_.items() if item not in scalars}
        evaluate = self._compile(path, expression, {}, readers)
        reductions = ReductionScheduleSpec(tuple(lowering.found_.values()), parameters)

        def read_expression(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            values = np.broadcast_to(np.asarray(evaluate(t, x, reductions.extend(t, x, p), done)), (size,))
            return ReductionKernel._reduce_values(values, observable.reduce)

        return read_expression

    @staticmethod
    def _get_observed_getter(kernel: LawKernel, observable: ObservableSpec) -> ObservableFunction:
        """Return the getter of a reduction of a field expression with field operators.

        The getter reads from its kernel.
        """

        def read_field(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del done
            _, _, values = kernel.run(RunStateSpec(t, x, p)).values_
            return ReductionKernel._reduce_values(values[0], observable.reduce)

        return read_field

    def _get_agent_getter(self, group: str, path: str, observable: ObservableSpec) -> ObservableFunction:
        """Return the getter of a reduction over an agent group.

        The agents of the evaluation give the values.
        """
        if not any(spec.path == group for spec in self.problem.agents) and not self.problem.partition_problems:
            error_msg = f'ObservableSpec {path} reduces {group}, which is neither an element group nor an agent group.'
            raise ValidationError(error_msg)
        objects = {name for spec in self.problem.agents if spec.path == group for name in spec.objects}

        def read_agents(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del t, x, p
            agents_of = done['__agents__']
            if not callable(agents_of):
                error_msg = f'The agents of an evaluation are given by a function, not a {type(agents_of).__name__}.'
                raise TypeError(error_msg)
            found = agents_of(group)
            agents: list[object] = list(found) if isinstance(found, list | tuple) else []
            values: list[object]
            if observable.variable is not None:
                values = [getattr(agent, observable.variable) for agent in agents]
            elif callable(observable.expression):
                values = [observable.expression(agent) for agent in agents]
            else:
                values = agents
            if observable.variable in objects:
                return ReductionKernel._reduce_values(
                    np.fromiter(values, dtype=object, count=len(values)),
                    observable.reduce,
                )
            return ReductionKernel._reduce_values(values, observable.reduce)

        return read_agents

    @staticmethod
    def _get_operator_getters(problem: OperatorProblemSpec) -> dict[str, ObservableFunction]:
        """Return the getters of the observables of an operator problem.

        The energy is included, and the purity when the problem is open. An observable gives an operator, a Sage
        expression of local operators, as a function of the namespace of its system, of its element, or of every member
        of its group, or names a local operator of its element or of every member of its group with `variable`, such as
        `ObservableSpec('z', element='q', variable='z')`. Its value is the expectation of the operator, reduced over the
        members of a group.

        Args:
            problem:
                The operator problem.

        Returns:
            getters:
                The getters by observable path.

        Raises:
            ValidationError:
                Two observables share a path, or an observable does not give an operator.
        """
        getters: dict[str, ObservableFunction] = {}
        for scope, observable in problem.observables:
            label = observable.label if observable.element is None else f'{observable.element}.{observable.label}'
            path = ObservableEvaluator._join(scope, label)
            if path in getters:
                error_msg = f'Two observables have the path {path}. Give observables of one system distinct labels.'
                raise ValidationError(error_msg)
            if observable.group is not None:
                group = ObservableEvaluator._join(scope, observable.group)
                getters[path] = ObservableEvaluator._get_operator_member_getter(problem, path, observable, group)
                continue
            if observable.element is not None and observable.variable is not None:
                element = ObservableEvaluator._join(scope, observable.element)
                terms = ObservableEvaluator._get_local_terms(problem, element, observable.variable)[0]
                getters[path] = ObservableEvaluator._get_expectation_getter(problem, terms)
                continue
            expression = observable.expression
            value = expression(problem.namespaces[scope]) if callable(expression) else expression
            terms = ObservableEvaluator._get_terms_of(problem, ObservableEvaluator._get_operator_of(path, value), path)
            getters[path] = ObservableEvaluator._get_expectation_getter(problem, terms)
        getters.setdefault('energy', ObservableEvaluator._get_expectation_getter(problem, problem.generator))
        if not problem.pure:
            getters.setdefault('purity', ObservableEvaluator._get_purity_getter(problem))
        return getters

    @staticmethod
    def _get_local_terms(problem: OperatorProblemSpec, path: str, name: str) -> list[tuple[OperatorKernel, ...]]:
        """Return the terms of the local operator of a name of every member.

        The members are those of an operator element or group.
        """
        lowering = ObservableEvaluator._get_lowering_of(problem)
        subsystems = [item for key, item in lowering.subsystems.items() if key == path or key.startswith(f'{path}[')]
        if not subsystems:
            error_msg = f'Observable of {path} reads {path}, which is not an operator element or group of the system.'
            raise ValidationError(error_msg)
        return [
            ObservableEvaluator._get_terms_of(problem, local_operator(item.symbol, SR.symbol(name)), path)
            for item in subsystems
        ]

    @staticmethod
    def _get_lowering_of(problem: OperatorProblemSpec) -> OperatorLowering:
        """Return the lowering of the operators of an operator problem.

        A problem without one raises an error.

        Args:
            problem:
                The operator problem.

        Returns:
            lowering:
                The lowering of its operators.

        Raises:
            ValidationError:
                The problem holds no lowering of its operators.
        """
        if problem.lowering is None:
            error_msg = 'The operator problem holds no lowering of its operators. Compile it with OperatorCompiler.'
            raise ValidationError(error_msg)
        return problem.lowering

    @staticmethod
    def _get_terms_of(problem: OperatorProblemSpec, operator: Expression, path: str) -> tuple[OperatorKernel, ...]:
        """Return the terms of an operator of an operator problem, lowered by its lowering.

        Args:
            problem:
                The operator problem.

            operator:
                The operator.

            path:
                The path of the observable the operator belongs to.

        Returns:
            terms:
                The terms of the operator.
        """
        return ObservableEvaluator._get_lowering_of(problem).run((operator, path)).kernels_

    @staticmethod
    def _get_operator_member_getter(
        problem: OperatorProblemSpec,
        path: str,
        observable: ObservableSpec,
        group: str,
    ) -> ObservableFunction:
        """Return the getter of an operator observable evaluated member by member.

        The values are reduced. The observable gives the operator of a member as a function of its namespace, or names a
        local operator of every member with `variable`.
        """
        if group not in problem.members:
            error_msg = f'Observable {path} reduces {group}, which is not an operator group of the system.'
            raise ValidationError(error_msg)
        expression = observable.expression
        if observable.variable is not None:
            terms = ObservableEvaluator._get_local_terms(problem, group, observable.variable)
        elif callable(expression):
            terms = [
                ObservableEvaluator._get_terms_of(
                    problem,
                    ObservableEvaluator._get_operator_of(path, expression(member)),
                    path,
                )
                for member in problem.members[group]
            ]
        else:
            error_msg = f'Observable {path} of an operator group needs a function of a member namespace or a variable.'
            raise ValidationError(error_msg)

        def get(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del done
            values = np.asarray([problem._expect(item, t, x, p) for item in terms])
            return ReductionKernel._reduce_values(values, observable.reduce)

        return get

    @staticmethod
    def _get_operator_of(path: str, value: object) -> Expression:
        """Return the operator an observable expression of an operator system gives.

        It first checks that it is an operator.
        """
        if not isinstance(value, Expression) or not has_operator(value):
            error_msg = f'Observable {path} of an operator system gives {value!r}, but it must give an operator.'
            raise ValidationError(error_msg)
        return value

    @staticmethod
    def _get_expectation_getter(problem: OperatorProblemSpec, terms: Sequence[OperatorKernel]) -> ObservableFunction:
        """Return the getter of the expectation value of an operator.

        It is taken at the state of an operator problem.
        """

        def get(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del done
            return problem._expect(terms, t, x, p)

        return get

    @staticmethod
    def _get_purity_getter(problem: OperatorProblemSpec) -> ObservableFunction:
        """Return the getter of the purity `Tr(rho^2)` of the state of an operator problem."""

        def get(t: float, x: Array, p: Array, done: Mapping[str, object]) -> object:
            del t, p, done
            state = problem._get_state_of(x)
            return 1.0 if problem.pure else float(np.sum(np.abs(state) ** 2))

        return get

    def evaluate(self, t: float, x: Array, p: Array, agents_of: AgentsFunction | None = None) -> dict[str, object]:
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
        done: dict[str, object] = {'__agents__': agents_of or (lambda group: None)}
        for path, get in self.getters_.items():
            done[path] = get(t, x, p, done)
        del done['__agents__']
        return done

    @staticmethod
    def _create_from_done(name: str) -> Callable[[Array, Array, Mapping[str, object]], object]:
        """Return a reader of an observable computed earlier in the same frame."""
        return lambda x, p, done: done[name]

    @staticmethod
    def _create_from_members(
        read: Callable[[Array, Array], Array],
    ) -> Callable[[Array, Array, Mapping[str, object]], object]:
        """Return a reader of one member variable over a group."""
        return lambda x, p, done: read(x, p)

    @staticmethod
    def _read_of(read: ReadSpec) -> Callable[[Array, Array], Array]:
        """Return a reader of the values a reduction read stands for.

        It reads them from the state or the extended parameters.
        """
        return ObservableEvaluator._create_from_vector(read.is_state, read.index)

    @staticmethod
    def _create_from_offset(
        column: Callable[[Array, Array], Array],
        read: NeighborReadSpec,
    ) -> Callable[[Array, Array], Array]:
        """Return a reader of a member column at an offset from every member."""
        return lambda x, p: read.read(column(x, p), p)

    @staticmethod
    def _create_from_vector(is_state: bool, indices: int | Indices) -> Callable[[Array, Array], Array]:
        """Return a reader of components of the state or the parameter vector."""
        return lambda x, p: (x if is_state else p)[indices]

    @staticmethod
    def _get_renewal_counter(counts: npt.NDArray[np.intp]) -> ObservableFunction:
        """Return the getter of the number of renewals of a renewal law.

        It counts over all members.
        """

        def count_renewals(t: float, x: Array, p: Array, done: Mapping[str, object]) -> float:
            del t, p, done
            return float(np.sum(x[counts]))

        return count_renewals

    @staticmethod
    def _get_joined_complex(expression: object, owner: str) -> Expression:
        """Return an expression as its real part plus `i` times its imaginary part.

        Both parts are real.

        Args:
            expression:
                The expression, whose symbols are all real.

            owner:
                The owner of the expression, used in error messages.

        Returns:
            expression:
                The expression with every complex function written out, ready to print with complex numbers.

        Raises:
            ValidationError:
                The expression cannot be split into real and imaginary parts.
        """
        real, imaginary = LowerComplexPass._split_complex(expression, owner)
        return real if LowerComplexPass._is_zero(imaginary) else real + I * imaginary

    @staticmethod
    def _get_entry_of(problem: BaseProblemSpec, name: str) -> EntrySpec | None:
        """Return the entry of a state or parameter of a problem by name, or `None`."""
        for layout in (problem.states, problem.parameters):
            if layout.has(name):
                return layout.get_entry(name)
        return None

    @staticmethod
    def _get_tree(items: Mapping[str, object], **root: object) -> SimpleNamespace:
        """Return nested namespaces from paths such as `forest/foxes.n` mapped to values.

        Extra values go at the root.

        Args:
            items:
                The values, by their path from the root.

            **root:
                The extra values of the root namespace, such as the time.

        Returns:
            namespace:
                The namespace of every element and nested system, with the extra values.
        """
        systems: dict[str, dict[str, object]] = {}
        elements: dict[str, dict[str, object]] = {}
        for path, value in items.items():
            head, slash, rest = path.partition('/')
            if slash:
                systems.setdefault(head, {})[rest] = value
                continue
            element, _, variable = path.partition('.')
            elements.setdefault(element, {})[variable] = value
        namespaces = {name: SimpleNamespace(**values) for name, values in elements.items()}
        namespaces.update({name: ObservableEvaluator._get_tree(values) for name, values in systems.items()})
        return SimpleNamespace(**namespaces, **root)

    @staticmethod
    def _get_namespace_of(problem: BaseProblemSpec, scope: str = '') -> SimpleNamespace:
        """Return the namespace of the states and parameters of a system.

        They are those of its single elements and interactions. The namespace nests the symbols by their dotted names,
        so `s.plant.x` is the state `plant.x`, and `s.forest.foxes.n` the state `forest/foxes.n`. It also holds the time
        `s.t`.

        Args:
            problem:
                The compiled problem.

            scope:
                The path of the nested system whose symbols the namespace holds, empty for the root.

        Returns:
            namespace:
                The namespace.
        """
        prefix = f'{scope}/' if scope else ''
        items: dict[str, object] = {
            name[len(prefix) :]: problem.symbols[name]
            for name in problem.symbols
            if name.startswith(prefix)
            and (entry := ObservableEvaluator._get_entry_of(problem, name)) is not None
            and entry.members is None
        }
        return ObservableEvaluator._get_tree(items, t=problem.time)

    @staticmethod
    def _join(scope: str, path: str) -> str:
        """Return a path relative to a system as a path from the root."""
        return f'{scope}/{path}' if scope else path

    @staticmethod
    def _is_lowerable(scope: str, observable: ObservableSpec, nodes: dict[str, NodeSpec]) -> bool:
        """Check whether laws can read an observable at the current state.

        They read it as an expression or a symbolic reduction.
        """
        if observable.group is None:
            return True
        node = nodes.get(ObservableEvaluator._join(scope, observable.group))
        return node is not None and not node.is_agent and isinstance(observable.reduce, str)

    @staticmethod
    def _get_lowerable_observables(system: CompositeSystem, nodes: list[NodeSpec]) -> Declared:
        """Return every observable of a system tree that laws can read, by path.

        Laws read them at the current state. An expression of a system's symbols is lowerable when every observable
        declared before it in its system is.
        """
        by_path = {node.path: node for node in nodes}
        lowerable: Declared = {}
        blocked: set[str] = set()
        for scope, observable in BaseCompiler._get_observables_of(system):
            if scope in blocked or not ObservableEvaluator._is_lowerable(scope, observable, by_path):
                blocked.add(scope)
                continue
            lowerable[ObservableEvaluator._join(scope, observable.label)] = (scope, observable)
        return lowerable

    @staticmethod
    def _get_held_reductions_of(problem: BaseProblemSpec) -> list[tuple[str, '_HeldReductionSpec']]:
        """Return the reductions a co-simulation computes at synchronisation points.

        They come with where it holds them, and they are computed at every such point. A law of one partition reads a
        reduction of the agents of another, such as `count(s.farmers)`, through an element that holds its value in the
        state `value` between synchronisation points.

        Args:
            problem:
                The problem.

        Returns:
            reductions:
                The path of every element that holds a reduction, with the reduction.
        """
        return [
            (element.path, reduction)
            for element in problem.elements
            if isinstance(element.element, _HeldElement) and (reduction := element.element.reduction) is not None
        ]
