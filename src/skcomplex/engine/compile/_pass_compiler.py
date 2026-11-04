"""Compile a system through the passes of a compilation, after validating it against what they need."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import heapq
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from shapely import Polygon
from sklearn.base import clone

from ...base import BaseConfig
from ...core.elements import FieldElement, GroupElement
from ...core.interactions import (
    ConnectionInteraction,
    HyperedgeInteraction,
    NeighborInteraction,
    SatisfiabilityInteraction,
)
from ...core.laws import BilinearLaw, EncounterLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw, SampleLaw
from ...core.laws.utils import COORDINATE_PREFIX, cell_operations_in, hyperedges_in, neighborhoods_in, reductions_in
from ...core.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    CayleySpace,
    CellComplexSpace,
    CofaceNeighborhood,
    ConeNeighborhood,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    PhaseSpace,
    StaticTime,
)
from ...core.specs import CensusRecordSpec, CensusSpec, NodeSpec
from ...core.system import System
from ...errors import (
    AlgebraicStateError,
    ParameterError,
    PhaseSpaceTooLargeError,
    PipelineError,
    SkcomplexError,
    SolverError,
    SpaceError,
    SpaceTooLargeError,
    ValidationError,
)
from ...specs import (
    EntrySpec,
    IssueSpec,
    NeighborReductionSpec,
    ObservableSpec,
    TopologyRequestSpec,
    WienerProcessSpec,
)
from ...specs._held_reduction_spec import _HeldReductionSpec
from ...specs._issue_spec import _ERROR
from ...types import Array, DomainKind, FormKind, NoiseKind, ProcessKey, VariableKind, VectorFunction
from .._utils import declarations_of, default_of, symbol
from ..backends import BaseBackend
from ..problems import BaseProblemSpec, PartitionSpec, ProblemSpec
from ..problems._neighborhood_kernel import NeighborhoodKernel
from ..problems._topology_kernel import TopologyKernel
from ._base import _FIXES, BaseCompiler, BasePass
from ._compilation import Compilation
from ._field_lowering import FieldLowering
from ._learning_compiler import _LearningCompiler
from ._observables import _check_held_reads, _held_observables, _with_held_observables, _with_held_reductions
from .passes._cells import DEGREES
from .passes._compile_encounters import CompileEncountersPass
from .passes._conserved_noise import ConservedNoisePass
from .passes._core import _PASSES
from .passes._fractional_form import FractionalFormPass
from .passes._lattices import LatticeProblemPass
from .passes._layout import LayoutPass
from .passes._noise import NoisePass
from .passes._operator_compiler import OperatorCompiler
from .passes._orders import OrdersPass
from .passes._reduction_lowering import ReductionLowering

_COVARIANCE_TOLERANCE = 1e-12
_PROBABILITY_TOLERANCE = 1e-9
_LISTED_ENCOUNTERS = 3
_COLLISION_AXES = 2
_TIME_CLASSES: dict[type, str] = {
    ContinuousTime: 'ContinuousTime()',
    DiscreteTime: 'DiscreteTime()',
    StaticTime: 'StaticTime()',
}
_OWN_NEIGHBORHOOD = (LatticeSpace, NetworkSpace, CayleySpace, MeshSpace)
_REPLACEMENTS = {
    DomainKind.CATEGORICAL: 'a jump, transition, or update law',
    DomainKind.INTEGER: 'a jump or update law',
    DomainKind.BOOLEAN: 'a jump or update law',
    DomainKind.HILBERT: 'a generator or dissipator law',
}
_PARTITIONS = ('plan', 'subsystems')
POLYGON_COORDINATES = 2
_PLANNING = frozenset({'nodes', 'census', 'plan'})
_FIELDS = (
    'system',
    'backend',
    'solver',
    'owners',
    'held',
    *(name for name in Compilation.__annotations__ if name != 'done'),
)


class PassCompiler(BaseCompiler):
    """General compiler of a system, which runs passes over a compilation in the order of the fields they share.

    `prepare(system)` validates the system: it compares the requirements every law declares with the states it
    targets, the space it lives on, and the system's time, checks that one solver runs the laws of every partition,
    and checks the solver and the backend against their declared support. A state of a fractional order needs an order
    in `(0, 1)`, continuous time, and a solver and a backend that integrate fractional orders. It reports every issue
    at once in `report_`, each with its path and fix, and keeps the compilation planned so far in `compilation_`, which
    `run` continues from. Elements and groups are compiled once each, whatever the size of a group. Laws written for one
    member are evaluated for every member, and pair interactions are evaluated for every pair within their cutoff.

    Args:
        passes:
            The passes, in the order they are registered, or `None` for every pass of the library. The compiler runs
            copies of them.

        backend:
            The backend instance, or `None` for `NumbaBackend` on systems whose groups hold at least 1000 members in
            total and `NumpyBackend` on the others.

        solver:
            The solver the problem runs on, or `None` for the default solver of its form.

        observables:
            The observables to compute at every frame, labelled by their path from the root.

        partition:
            `'plan'` to compile one problem, or one partition per planned subsystem when no single form runs the
            system, or `'subsystems'` to compile one partition for the elements of the root and one per child
            system, so a co-simulation can advance them with separate solvers.

    Attributes:
        passes_:
            The copies of the passes the compiler runs.

        compilation_:
            The compilation planned so far, or `None` when the system tree cannot be flattened or trains a nested
            system.

        learning_:
            The compiler of a system that holds a `TrainingInteraction`, which compiles its nested system into the
            inner problem of a `LearningProblemSpec`, or `None` for other systems.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import PassCompiler
        >>> from skcomplex.models.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> system = System(
        ...     elements=[PopulationElement('rabbits', n=10)],
        ...     interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
        ... )
        >>> compiler = PassCompiler().check().prepare(system)
        >>> compiler.report_.ok
        True
        >>> compiler.run().problem_.equations
        {'rabbits.n': n_rabbits*r_growth}
    """

    passes_: tuple[BasePass, ...]
    compilation_: Compilation | None
    learning_: _LearningCompiler | None

    def __init__(
        self,
        passes: Sequence[BasePass] | None = None,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[ObservableSpec] = (),
        partition: str = 'plan',
    ) -> None:
        super().__init__(backend, solver, observables)
        self.passes = passes
        self.partition = partition

    def _check(self) -> None:
        """Raise an error for passes that are not passes or a partition that is not one of the choices.

        Raises:
            ValidationError:
                A setting is invalid.
        """
        super()._check()
        if self.passes is not None and not all(isinstance(item, BasePass) for item in self.passes):
            error_msg = 'The passes hold an object that is not a pass. Pass instances of BasePass.'
            raise ValidationError(error_msg)
        if self.partition not in _PARTITIONS:
            error_msg = f'The partition is {self.partition!r}. Choose one of {", ".join(map(repr, _PARTITIONS))}.'
            raise ValidationError(error_msg)

    def _issues(self, system: System) -> list[IssueSpec]:
        """Return every issue of a system, of its solver, and of its backend, planning the compilation as it goes.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues, in the order they were found.
        """
        passes = self.passes if self.passes is not None else [item() for item in _PASSES]
        self.passes_ = tuple(clone(item) for item in passes)
        self.learning_ = None
        if _LearningCompiler._find_training(system) is not None:
            inner = PassCompiler(passes=self.passes)
            self.learning_ = _LearningCompiler(inner, self.backend, self.solver).check().prepare(system)
            self.compilation_ = None
            return list(self.learning_.report_.issues)
        self.compilation_, issues = self._counted(system)
        if self.compilation_ is None or self.compilation_.census is None:
            return issues
        issues = self._issues_of(self.compilation_)
        if not any(issue.severity == _ERROR for issue in issues):
            issues += self._structure_issues(self.compilation_)
        if not any(issue.severity == _ERROR for issue in issues):
            issues += self._plan_issues(self.compilation_)
        return issues

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblemSpec:
        """Compile the prepared system into one problem, or one partition per planned subsystem or per child system.

        Args:
            initial_state:
                Initial values that replace those of the components, by the dotted name of the state.

        Returns:
            problem:
                The compiled problem.
        """
        if self.learning_ is not None:
            return self.learning_.run(initial_state).problem_
        compilation = self.compilation_
        if compilation is None:
            error_msg = 'The system stopped compiling before its census. Prepare the compiler on a valid system.'
            raise PipelineError(error_msg)
        if self.partition == 'subsystems':
            return self._with_initial_state(self._subsystem_problem(compilation), initial_state)
        problem = self._planned_problem(compilation)
        _check_held_reads(problem, list(compilation.held))
        return self._with_initial_state(problem, initial_state)

    @staticmethod
    def _with_initial_state(problem: BaseProblemSpec, initial_state: Mapping[str, object]) -> BaseProblemSpec:
        """Return a problem whose initial state takes given values of some states.

        Args:
            problem:
                The compiled problem.

            initial_state:
                The initial values by dotted state name.

        Returns:
            problem:
                The problem with the new initial state, or the same problem when no value is given.

        Raises:
            ValidationError:
                A name is not a state of the problem, or a value has the wrong shape.
        """
        if not initial_state:
            return problem
        names = problem.states.names
        unknown = set(initial_state) - set(names)
        if unknown:
            error_msg = (
                f'The initial state names {", ".join(sorted(unknown))}, which are not states of the system. Use the '
                f'dotted names of states, such as {", ".join(list(names)[:3])}.'
            )
            raise ValidationError(error_msg)
        state = problem.initial_state.copy()
        for name, value in initial_state.items():
            entry = problem.states.entry(name)
            try:
                state[entry.indices] = entry.encode(value)
            except ValueError as error:
                error_msg = f'The initial value of {name} does not fit its shape {entry.shape}: {error}'
                raise ValidationError(error_msg) from error
        return dataclasses.replace(problem, initial_state=state)

    def _new_compilation(self, system: System, held: tuple[str, ...] = ()) -> Compilation:
        """Return a new compilation of a system, with no pass run yet."""
        compilation = Compilation(system, self.backend, self.solver, held=held)
        compilation.done = set()
        return compilation

    @staticmethod
    def _present(compilation: Compilation) -> set[str]:
        """Return the fields a compilation holds: its inputs and the fields set so far."""
        return {name for name in _FIELDS if getattr(compilation, name) is not None}

    @staticmethod
    def _fork(compilation: Compilation, owners: frozenset[str]) -> Compilation:
        """Return the compilation of a partition, which shares the nodes, the census, and the plan."""
        fork = Compilation(compilation.system, compilation.backend, compilation.solver, owners, compilation.held)
        fork.nodes, fork.census, fork.plan = compilation.nodes, compilation.census, compilation.plan
        fork.done = {
            item
            for item in Compilation._required(compilation.done, 'done')
            if isinstance(item, BasePass) and set(item._get_writes()) <= _PLANNING
        }
        return fork

    def _run_passes(self, compilation: Compilation, until: str = 'problem') -> Compilation:
        """Run the passes a field needs that the compilation has not run yet, in the order of their fields.

        Args:
            compilation:
                The compilation, which the passes change in place.

            until:
                The field to produce. The passes that neither write it nor write a field that those passes read do not
                run.

        Returns:
            compilation:
                The same compilation, holding the field.

        Raises:
            PipelineError:
                The passes cannot be ordered.
        """
        needed = self._needed(self.passes_, until)
        done = Compilation._required(compilation.done, 'done')
        for item in self._ordered(self.passes_, self._present(compilation)):
            if item in needed and item not in done:
                item.check().prepare(compilation).run()
                done.add(item)
        return compilation

    def _compiled(self, compilation: Compilation) -> ProblemSpec:
        """Return the problem of a compilation, running every pass it has not run yet."""
        return Compilation._required(self._run_passes(compilation).problem, 'problem')

    def _partitioned(self, compilation: Compilation, plan: list[tuple[str, frozenset[str]]]) -> ProblemSpec:
        """Return the problem of a system split into partitions, each compiled with the laws of its owners."""
        partitions = []
        for name, owners in plan:
            problem = self._compiled(self._fork(compilation, owners))
            owned = [entry.indices for entry in problem.states.entries if entry.name.rpartition('.')[0] in owners]
            indices = np.concatenate([np.arange(item.start, item.stop) for item in owned]) if owned else np.zeros(0)
            partitions.append(PartitionSpec(name, problem, indices.astype(np.intp), owners))
        first = partitions[0].problem
        equations = {
            key: value
            for item in partitions
            for key, value in item.problem.equations.items()
            if key.partition('.')[0] in item.owners
        }
        symbols = {
            key: item.problem.equation_symbols[key]
            for item in partitions
            for key in item.problem.equations
            if key in equations
        }
        return ProblemSpec(
            form=FormKind.COUPLED,
            states=first.states,
            parameters=first.parameters,
            initial_state=first.initial_state,
            initial_parameters=first.initial_parameters,
            time=first.time,
            symbols=first.symbols,
            equations=equations,
            equation_symbols=symbols,
            has_callables=any(item.problem.has_callables for item in partitions),
            has_group_sums=any(item.problem.has_group_sums for item in partitions),
            elements=first.elements,
            interactions=first.interactions,
            system=compilation.system,
            agents=tuple(agents for item in partitions for agents in item.problem.agents),
            agent_space=next((item.problem.agent_space for item in partitions if item.problem.agents), None),
            partitions=tuple(partitions),
            algebraic=self._partition_algebraic(partitions),
        )

    @staticmethod
    def _partition_algebraic(partitions: Sequence[PartitionSpec]) -> VectorFunction | None:
        """Return the function that sets the algebraic states of every partition, or `None` without them."""
        functions = [item.problem.algebraic for item in partitions if item.problem.algebraic is not None]
        if not functions:
            return None

        def solve(t: float, x: Array, p: Array) -> Array:
            state = x
            for function in functions:
                state = function(t, state, p)
            return state

        return solve

    def _planned_problem(self, compilation: Compilation) -> ProblemSpec:
        """Return the problem of a validated compilation, one problem or one partition per planned subsystem.

        The owners of a partition are the elements and groups of its subsystem whose lowered laws it holds.
        """
        plan = Compilation._required(compilation.plan, 'plan')
        if len(plan.partitions) <= 1:
            return self._compiled(compilation)
        kinds = Compilation._required(self._run_passes(compilation, until='kinds').kinds, 'kinds')
        owners = [(item.name, frozenset(owner for owner in kinds if item.holds(owner))) for item in plan.partitions]
        return self._partitioned(compilation, owners)

    def _subsystem_problem(self, compilation: Compilation) -> ProblemSpec:
        """Return the problem of a system as one partition for the elements of its root and one per child system."""
        system = compilation.system
        elements = {node.path for node in Compilation._required(compilation.nodes, 'nodes') if not node.interaction}
        own = frozenset(path for path in elements if '/' not in path)
        plan = [(system.label, own)] if own else []
        for child in system.systems:
            plan.append((child.label, frozenset(path for path in elements if path.startswith(f'{child.label}/'))))
        return self._partitioned(compilation, plan)

    @staticmethod
    def _compiler_issue(system: System, compiler: str) -> IssueSpec:
        """Return the issue of a system that another compiler compiles, naming that compiler."""
        message = f'The system {system.label} is compiled by {compiler}, not by PassCompiler.'
        return IssueSpec('compiler', system.label, message, f'Pass `compiler={compiler}()`.')

    def _counted(self, system: System) -> tuple[Compilation | None, list[IssueSpec]]:
        """Return the compilation of a system through the census of its laws, or the issues that stop it before.

        The compilation stops after flattening for a system that another compiler compiles. When laws read reductions of
        agents, the compilation starts again from a system tree that holds each of them in an element.
        """
        compilation = self._new_compilation(system)
        try:
            self._run_passes(compilation, until='nodes')
        except PhaseSpaceTooLargeError as error:
            fix = 'Use fewer points on the axes of the phase space, or raise its limit.'
            return None, [IssueSpec('space-size', system.label, str(error), fix, error=PhaseSpaceTooLargeError)]
        except SkcomplexError as error:
            fix = 'Pass elements, groups, interactions, and systems as children.'
            return None, [self._issue_of(error, system.label, 'structure', fix)]
        issues = (
            [self._compiler_issue(system, 'OperatorCompiler')]
            if OperatorCompiler._contains_operator(system)
            else self._port_issues(Compilation._required(compilation.nodes, 'nodes'))
        )
        if issues:
            return compilation, issues
        held = _held_observables(system, Compilation._required(compilation.nodes, 'nodes'))
        if held:
            tree, paths = _with_held_observables(system, held)
            compilation = self._new_compilation(tree, held=tuple(paths))
        self._run_passes(compilation, until='census')
        try:
            reductions = self._held_reductions(Compilation._required(compilation.census, 'census'), symbol('t'))
        except ValidationError as error:
            fix = 'Reduce an expression of the agents of one group.'
            return None, [self._issue_of(error, system.label, 'reduction', fix)]
        if not reductions:
            return compilation, []
        tree = _with_held_reductions(compilation.system, reductions)
        compilation = self._new_compilation(tree, held=compilation.held)
        return self._run_passes(compilation, until='census'), []

    def _structure_issues(self, compilation: Compilation) -> list[IssueSpec]:
        """Return the issue of equations that cannot be matched to their unknowns, lowering the laws as needed."""
        if not self._algebraic_names(compilation):
            return []
        try:
            self._run_passes(compilation, until='lowered')
            return OrdersPass._structure_issues(compilation)
        except SkcomplexError as error:
            fix = 'Correct the component as the message describes.'
            return [self._issue_of(error, compilation.system.label, 'value', fix)]

    @staticmethod
    def _algebraic_names(compilation: Compilation) -> set[str]:
        """Return the dotted names of the algebraic states of the elements and groups of a system."""
        return {
            f'{node.path}.{name}'
            for node in Compilation._required(compilation.nodes, 'nodes')
            if not node.interaction
            for name, declaration in declarations_of(node.component_type, VariableKind.STATE).items()
            if declaration.algebraic
        }

    @staticmethod
    def _port_state_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the variables of declared ports that have no algebraic state."""
        issues = []
        for node in nodes:
            ports = {} if node.interaction else getattr(node.component_type, 'ports', {})
            declarations = {} if node.interaction else declarations_of(node.component_type, VariableKind.STATE)
            for port, connector in ports.items():
                for variable in (connector.potential, connector.flow):
                    name = f'{port}_{variable}'
                    declaration = declarations.get(name)
                    if declaration is None or not declaration.algebraic:
                        message = (
                            f'The port {port} of {node.path} has the connector {connector.name}, but the element '
                            f'declares no algebraic state {name}.'
                        )
                        fix = f'Declare {name} with `StateSpec(algebraic=True)`.'
                        issues.append(IssueSpec('port-states', node.path, message, fix))
        return issues

    @staticmethod
    def _connector_issue(
        node: NodeSpec,
        connector_names: Mapping[str, str | None],
    ) -> IssueSpec | None:
        """Return the issue of a connection whose ports are not all ports of one connector, or `None`."""
        missing = [port for port, name in connector_names.items() if name is None]
        if missing:
            message = (
                f'The connection {node.path} names {", ".join(missing)}, which no single element declares as a port.'
            )
            fix = 'Connect declared ports of single elements, as listed in the class variable `ports` of their types.'
            return IssueSpec('port-connector', node.path, message, fix)
        if len(set(connector_names.values())) <= 1:
            return None
        listed = ', '.join(f'{port} of the connector {name}' for port, name in connector_names.items())
        message = f'The connection {node.path} joins {listed}.'
        return IssueSpec('port-connector', node.path, message, 'Connect ports of one connector.')

    @staticmethod
    def _port_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the issues of declared ports and of the connections between them.

        Args:
            nodes:
                The elements, groups, and interactions of the system tree.

        Returns:
            issues:
                The ports without their states, the connections of ports of different connectors, and the ports in more
                than one connection.
        """
        issues = PassCompiler._port_state_issues(nodes)
        by_path = {node.path: node for node in nodes}
        connections: defaultdict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if not isinstance(node.component, ConnectionInteraction):
                continue
            names: dict[str, str | None] = {}
            for member, port in zip(node.component.members, node.component.ports, strict=True):
                path = f'{node.scope}{member}'
                target = by_path.get(path)
                ports = (
                    {} if target is None or target.members is not None else getattr(target.component_type, 'ports', {})
                )
                names[f'{path}.{port}'] = ports[port].name if port in ports else None
                connections[f'{path}.{port}'].append(node.path)
            issue = PassCompiler._connector_issue(node, names)
            issues += [] if issue is None else [issue]
        for port, paths in connections.items():
            if len(paths) > 1:
                message = f'The port {port} appears in the connections {", ".join(paths)}.'
                fix = 'List the ports of the node in one connection.'
                issues.append(IssueSpec('port-connections', paths[0], message, fix))
        return issues

    @staticmethod
    def _algebraic_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the laws other than equations that target an algebraic state."""
        algebraic = PassCompiler._algebraic_names(compilation)
        issues = []
        for record in Compilation._required(compilation.census, 'census').records:
            names = [] if isinstance(record.law, EquationLaw) else [item.name for item in record.targets]
            for name in names:
                if name in algebraic:
                    law = type(record.law).__name__
                    message = (
                        f'The {law} of {record.path} targets the algebraic state {name}, but the equations of the '
                        'system determine this state.'
                    )
                    fix = f'State the value of {name} with an `EquationLaw`, or declare it as a state of order 1.'
                    issues.append(IssueSpec('algebraic-law', record.path, message, fix, error=AlgebraicStateError))
        return issues

    def _plan_issues(self, compilation: Compilation) -> list[IssueSpec]:
        """Return the issues of the partitions of a system, lowering its laws as far as their classes decide."""
        try:
            self._run_passes(compilation, until='plan')
        except SkcomplexError as error:
            fix = 'Correct the component as the message describes.'
            return [self._issue_of(error, compilation.system.label, 'value', fix)]
        return list(Compilation._required(compilation.plan, 'plan').issues)

    @staticmethod
    def _issues_of(compilation: Compilation) -> list[IssueSpec]:
        """Return the issues of a compilation planned past the census."""
        system, solver = compilation.system, compilation.solver
        nodes = Compilation._required(compilation.nodes, 'nodes')
        census = Compilation._required(compilation.census, 'census')
        records = census.records
        issues = PassCompiler._census_issues(compilation)
        issues += PassCompiler._bound_issues(nodes)
        issues += PassCompiler._law_issues(records, FractionalFormPass._declared_times(system))
        issues += PassCompiler._algebraic_issues(compilation)
        issues += PassCompiler._equation_issues(compilation)
        issues += PassCompiler._satisfiability_issues(nodes)
        issues += PassCompiler._neighborhood_issues(records, nodes)
        issues += PassCompiler._around_issues(records)
        issues += PassCompiler._degree_issues(records)
        issues += PassCompiler._nonlocal_issues(records, nodes)
        issues += PassCompiler._coordinate_issues(records, nodes)
        issues += PassCompiler._space_issues(system, nodes)
        issues += PassCompiler._location_issues(nodes)
        issues += LatticeProblemPass._schedule_issues(compilation)
        issues += ConservedNoisePass._conserved_issues(compilation)
        issues += FractionalFormPass._order_issues(compilation)
        issues += PassCompiler._probability_issues(compilation)
        issues += PassCompiler._request_issues(compilation)
        if solver is not None:
            issues += PassCompiler._solver_issues(solver, census, system)
        issues += PassCompiler._noise_issues(compilation, records)
        return issues

    @staticmethod
    def _roles(passes: Sequence['BasePass']) -> dict[str, tuple[list[int], list[int], list[int]]]:
        """Return the producers, refiners, and readers of every field by the positions of their passes.

        A producer writes a field without reading it, a refiner reads and writes it, and a reader only reads it.
        """
        roles: dict[str, tuple[list[int], list[int], list[int]]] = defaultdict(lambda: ([], [], []))
        for index, item in enumerate(passes):
            for name in item._get_writes():
                roles[name][1 if name in item._get_reads() else 0].append(index)
            for name in item._get_reads():
                if name not in item._get_writes():
                    roles[name][2].append(index)
        return roles

    @staticmethod
    def _check_inputs(
        passes: Sequence['BasePass'],
        roles: dict[str, tuple[list[int], list[int], list[int]]],
        present: Collection[str],
    ) -> None:
        """Raise an error for a field read but neither present nor produced, or produced by two passes."""
        for name, (producers, refiners, readers) in roles.items():
            if len(producers) > 1:
                names = ' and '.join(type(passes[index]).__name__ for index in producers)
                error_msg = (
                    f'The passes {names} both produce the field {name!r}. Let one produce it and the others refine it.'
                )
                raise PipelineError(error_msg)
            if not producers and name not in present and (refiners or readers):
                index = (refiners or readers)[0]
                error_msg = (
                    f'The pass {type(passes[index]).__name__} reads the field {name!r}, which no pass produces and the '
                    'compilation does not hold. Add the pass that produces it.'
                )
                raise PipelineError(error_msg)

    @staticmethod
    def _edges(roles: dict[str, tuple[list[int], list[int], list[int]]]) -> dict[int, set[int]]:
        """Return the passes that must run after every pass: refiners after producers, readers after both."""
        edges: dict[int, set[int]] = defaultdict(set)
        for producers, refiners, readers in roles.values():
            for first in producers:
                edges[first] |= {*refiners, *readers}
            for first in refiners:
                edges[first] |= set(readers)
        return edges

    @staticmethod
    def _needed(passes: Sequence['BasePass'], field: str) -> set['BasePass']:
        """Return the passes that write a field and, in turn, the passes that write the fields they read."""
        writers: dict[str, list[BasePass]] = defaultdict(list)
        for item in passes:
            for name in item._get_writes():
                writers[name].append(item)
        needed: set[BasePass] = set()
        pending = [field]
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            for item in writers[name]:
                needed.add(item)
                pending += item._get_reads()
        return needed

    @staticmethod
    def _ordered(passes: Sequence['BasePass'], present: Collection[str] = ()) -> list['BasePass']:
        """Return passes in an order where every field is produced, then refined, then read.

        A pass that writes a field without reading it produces the field, and runs before the passes that refine it,
        which read and write it, and those run before the passes that only read it. Passes that the fields leave
        unordered run in the order they are given, so refiners of one field run in the order they are given.

        Args:
            passes:
                The passes, in the order they are registered.

            present:
                The fields the compilation already holds, which no pass needs to produce.

        Returns:
            passes:
                The passes in the order they run.

        Raises:
            PipelineError:
                A pass reads a field that no pass produces and the compilation does not hold, two passes produce one
                field, or the passes read each other's fields in a cycle.

        Examples:
            >>> from skcomplex.engine.compile import BasePass, PassCompiler
            >>> class LowerPass(BasePass):
            ...     def _get_reads(self):
            ...         return ('layout',)
            ...     def _get_writes(self):
            ...         return ('lowered',)
            >>> class DeclarePass(BasePass):
            ...     def _get_reads(self):
            ...         return ('nodes',)
            ...     def _get_writes(self):
            ...         return ('layout',)
            >>> ordered = PassCompiler._ordered([LowerPass(), DeclarePass()], present=['nodes'])
            >>> [type(item).__name__ for item in ordered]
            ['DeclarePass', 'LowerPass']
        """
        roles = PassCompiler._roles(passes)
        PassCompiler._check_inputs(passes, roles, present)
        edges = PassCompiler._edges(roles)
        waiting = [0] * len(passes)
        for targets in edges.values():
            for target in targets:
                waiting[target] += 1
        ready = [index for index, count in enumerate(waiting) if not count]
        heapq.heapify(ready)
        ordered: list[int] = []
        while ready:
            index = heapq.heappop(ready)
            ordered.append(index)
            for target in edges[index]:
                waiting[target] -= 1
                if not waiting[target]:
                    heapq.heappush(ready, target)
        if len(ordered) < len(passes):
            names = ', '.join(type(item).__name__ for index, item in enumerate(passes) if index not in set(ordered))
            error_msg = f'The passes {names} read each other\'s fields in a cycle. Split one of them.'
            raise PipelineError(error_msg)
        return [passes[index] for index in ordered]

    @staticmethod
    def _names(classes: tuple[type | None, ...]) -> str:
        """Return the names of classes joined by `or`, with `no space` for `None`."""
        return ' or '.join('no space' if item is None else item.__name__ for item in classes)

    @staticmethod
    def _layer_issues(node: NodeSpec, space: BaseSpace | None) -> list[IssueSpec]:
        """Return the issue of an interaction over neighbours that names a layer its network does not have."""
        layer = getattr(node.component, 'layer', None)
        labels = tuple(getattr(space, 'graphs_', (None,)))
        if layer is None or layer in labels:
            return []
        known = ', '.join(repr(label) for label in labels)
        message = (
            f'The interaction runs on the layer {layer!r}, which the network does not have. Its layers are {known}.'
        )
        fix = (
            f'Name one of the layers {known}, or add the layer with '
            f'`NetworkSpace(graph, layers=[LayerSpec({layer!r}, ...)])`.'
        )
        return [IssueSpec('binding', node.path, message, fix)]

    @staticmethod
    def _hyperedge_issues(node: NodeSpec, space: NetworkSpace, edges: NodeSpec | None) -> list[IssueSpec]:
        """Return the issues of a hyperedge interaction whose size, or whose edge group, its hyperedges do not fit.

        Args:
            node:
                The hyperedge interaction.

            space:
                The network of its nodes.

            edges:
                The group on edges its second role binds, or `None`.

        Returns:
            issues:
                The issue of a size no hyperedge has, or of the first two members of a hyperedge joined by no edge.
        """
        size = getattr(node.component, 'size', None)
        sizes = sorted({len(hyperedge) for hyperedge in space.hyperedges_})
        if size is not None and size not in sizes:
            known = ', '.join(map(str, sizes))
            message = (
                f'The interaction runs on hyperedges of size {size}, but the network has hyperedges of size {known}.'
            )
            fix = f'Give the interaction one of the sizes {known}, or add hyperedges of size {size}.'
            return [IssueSpec('binding', node.path, message, fix)]
        if edges is None:
            return []
        layer = getattr(edges.component, 'layer', None)
        graph = space.graphs_.get(layer)
        if graph is None:
            return []
        missing = [
            hyperedge[:2]
            for hyperedge in space.hyperedges_
            if (size is None or len(hyperedge) == size)
            and not graph.has_edge(*(list(space.graph.nodes)[index] for index in hyperedge[:2]))
        ]
        if not missing:
            return []
        message = (
            f'The edge group {edges.path} has no edge between the first two members of the hyperedges at the nodes '
            f'{missing[:3]}.'
        )
        fix = 'Order every hyperedge so an edge of the layer of the edge group joins its first two members.'
        return [IssueSpec('binding', node.path, message, fix)]

    @staticmethod
    def _binding_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the interactions over hyperedges, or over neighbours with an edge group, on a space without them.

        An interaction over neighbours of members of a space without edges runs only the laws of the members.
        """
        by_path = {node.path: node for node in nodes}
        issues = []
        for node in nodes:
            interaction = node.component
            if isinstance(interaction, NeighborInteraction):
                over = 'neighbors'
            elif isinstance(interaction, HyperedgeInteraction):
                over = 'hyperedges'
            else:
                continue
            roles = declarations_of(type(interaction), VariableKind.ROLE)
            labels = [getattr(interaction, name, None) for name in roles]
            bound = [
                by_path.get(f'{node.scope}{label}')
                for value in labels
                if value is not None
                for label in (value if isinstance(value, list | tuple) else [value])
            ]
            members = [item for item in bound if item is not None and getattr(item.component, 'on', None) != 'edges']
            target = next(iter(members), None)
            space = target.space if target is not None else node.space
            offered = (
                isinstance(space, LatticeSpace | NetworkSpace | CayleySpace)
                if over == 'neighbors'
                else bool(getattr(space, 'hyperedges_', ()))
            )
            if not offered and over == 'neighbors' and len(members) == len(bound):
                continue
            if not offered:
                needs = 'a lattice or network space' if over == 'neighbors' else 'a network space with hyperedges'
                message = f'The binding over {over} needs {needs}, but the bound members live on {space!r}.'
                issues.append(IssueSpec('binding', node.path, message, f'Give the bound group {needs}.'))
                continue
            if over == 'hyperedges' and isinstance(space, NetworkSpace):
                edges = next((item for item in bound if item is not None and item not in members), None)
                issues += PassCompiler._hyperedge_issues(node, space, edges)
                continue
            issues += PassCompiler._layer_issues(node, space)
        return issues

    @staticmethod
    def _shape_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the shaped parameters whose roles are not bound to groups."""
        groups = LayoutPass._group_sizes(nodes)
        issues = []
        for node in nodes:
            declarations = {
                **declarations_of(node.component_type, VariableKind.STATE),
                **declarations_of(node.component_type, VariableKind.PARAMETER),
            }
            for name, declaration in declarations.items():
                if declaration.shape is None:
                    continue
                try:
                    LayoutPass._role_shape(node, name, declaration, groups)
                except ValidationError as error:
                    path = f'{node.path}.{name}'
                    fix = f'Bind every role of the shape {declaration.shape} of {path} to a group.'
                    issues.append(IssueSpec('parameter-shape', path, str(error), fix, error=ParameterError))
        return issues

    @staticmethod
    def _bound_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the adaptive parameters whose initial values lie outside their bounds."""
        issues = []
        for node in nodes:
            for name, declaration in declarations_of(node.component_type, VariableKind.STATE).items():
                if declaration.bounds is None:
                    continue
                lower, upper = declaration.bounds
                value = np.asarray(LayoutPass._value_of(node, name), dtype=float)
                if np.all((value >= lower) & (value <= upper)):
                    continue
                path = f'{node.path}.{name}'
                message = (
                    f'The adaptive parameter {path} starts at {value.tolist()}, outside its bounds ({lower}, {upper}).'
                )
                fix = f'Give {path} an initial value inside its bounds ({lower}, {upper}).'
                issues.append(IssueSpec('parameter-bounds', path, message, fix, error=ParameterError))
        return issues

    @staticmethod
    def _space_issues(system: System, nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the visuals that need options their space lacks, and the spaces whose own check fails."""
        issues = []
        spaces: dict[int, tuple[str, BaseSpace]] = {}
        for node in nodes:
            space = getattr(node.component, 'space', None) or node.space
            visual = getattr(node.component, 'visual', None)
            if space is not None:
                spaces.setdefault(id(space), (node.path, space))
            if space is not None and 'image' in space._get_lacks() and getattr(visual, 'kind', None) == 'image':
                message = (
                    f'The image visual of {node.path} needs a regular lattice, but its space is {type(space).__name__}.'
                )
                default = getattr(space, 'default_visual_', None)
                fix = f'Use the {default!r} visual.' if default is not None else 'Use another visual.'
                issues.append(IssueSpec('visual-unsupported', node.path, message, fix))
        if system.space is not None:
            spaces.setdefault(id(system.space), (system.label, system.space))
        for path, space in spaces.values():
            try:
                space.check().prepare()
            except SpaceTooLargeError as error:
                fix = 'Ask for a smaller ball, or raise max_sites.'
                issues.append(IssueSpec('space-too-large', path, str(error), fix, error=SpaceTooLargeError))
            except ValidationError as error:
                fix = f'Give the {type(space).__name__} a shape, a boundary, and a geometry that it supports.'
                holder = path.rpartition('/')[0] or system.label
                issues.append(IssueSpec('space-invalid', holder, str(error), fix, error=SpaceError))
        return issues

    @staticmethod
    def _location_issue(path: str, group: GroupElement, size: int, sides: int) -> IssueSpec | None:
        """Return the first initial location of a group whose number of coordinates differs from its space's."""
        for index in range(size):
            location = group._get_member_location(index, size)
            length = POLYGON_COORDINATES if isinstance(location, Polygon) else np.asarray(location, dtype=object).size
            if length != sides:
                message = (
                    f'The initial location of member {index} of {path} has {length} coordinates, and its continuous '
                    f'space has {sides} dimensions.'
                )
                return IssueSpec('location-dimension', path, message, f'Give {sides} coordinates.', error=SpaceError)
        return None

    @staticmethod
    def _location_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the groups whose initial locations do not have one coordinate per dimension of their space."""
        issues = []
        for node in nodes:
            group, space = node.component, node.space
            if not isinstance(group, GroupElement) or group.location is None or not isinstance(space, ContinuousSpace):
                continue
            sides = len(tuple(space.extent)) or space.n_dim
            issue = PassCompiler._location_issue(node.path, group, node.members or 1, sides) if sides else None
            issues += [issue] if issue is not None else []
        return issues

    @staticmethod
    def _law_issues(records: list[CensusRecordSpec], times: dict[str, type | None]) -> list[IssueSpec]:
        """Return the laws whose space or time their requirements do not accept."""
        issues = []
        for record in records:
            time = times.get(record.path.rpartition('/')[0])
            law = record.law
            requires = law._get_requirements()
            name = f'{type(law).__name__} of {record.path}'
            if requires.space and not isinstance(record.space, requires.space):
                lives = type(record.space).__name__ if record.space is not None else 'no space'
                needs = PassCompiler._names(requires.space)
                message = f'The {name} needs a {needs}, but the component lives on {lives}.'
                issues.append(
                    IssueSpec('law-space', record.path, message, 'Give the system a space of a supported class.'),
                )
            if time is not None and requires.time and not issubclass(time, requires.time):
                needs = PassCompiler._names(requires.time)
                message = f'The {name} needs {needs}, but its system declares {time.__name__}.'
                classes = ' or '.join(f'`{_TIME_CLASSES[item]}`' for item in requires.time)
                fix = f'Declare {classes} for its system, or use laws that run in {time.__name__}.'
                issues.append(IssueSpec('law-time', record.path, message, fix))
        return issues

    @staticmethod
    def _equation_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the equations under static time with no target or with a target that is not a state."""
        if not isinstance(compilation.system.time, StaticTime):
            return []
        issues = []
        for record in Compilation._required(compilation.census, 'census').records:
            law = record.law
            if not isinstance(law, EquationLaw) or record.targets:
                continue
            problem = 'names no target' if law.target is None else f'names the target {law.target}, which is no state'
            message = f'The EquationLaw of {record.path} {problem}. An equation under static time solves for one state.'
            fix = 'Pass the symbol of a state of the system as `target`.'
            issues.append(IssueSpec('equation-target', record.path, message, fix))
        return issues

    @staticmethod
    def _satisfiability_issues(nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the satisfiability interactions that bind a state that is not Boolean."""
        by_path = {node.path: node for node in nodes}
        issues = []
        for node in nodes:
            component = node.component
            if not isinstance(component, SatisfiabilityInteraction):
                continue
            scope = node.path.rpartition('/')[0]
            role = getattr(component, component._get_role('target'))
            group = by_path.get(f'{scope}/{role}' if scope else role)
            declaration = (
                None
                if group is None
                else declarations_of(group.component_type, VariableKind.STATE).get(component.state)
            )
            if declaration is None or EntrySpec._domain_kind(declaration.domain) == DomainKind.BOOLEAN:
                continue
            message = (
                f'The SatisfiabilityInteraction {node.path} decides the state {component.state} of {role}, which is '
                'not Boolean.'
            )
            fix = 'Declare the state with `StateSpec(domain=FiniteEnumeratedSet([0, 1]))`.'
            issues.append(IssueSpec('law-domain', node.path, message, fix))
        return issues

    @staticmethod
    def _reductions_of(value: object) -> list[NeighborReductionSpec]:
        """Return every neighbourhood reduction of a law value, nested ones included."""
        found = []
        stack = neighborhoods_in(value)
        while stack:
            reduction = stack.pop()
            found.append(reduction)
            stack += neighborhoods_in(reduction.argument)
            stack += neighborhoods_in(reduction.weight) if reduction.weight is not None else []
        return found

    @staticmethod
    def _degree_problem(name: str, degree: int, top: int) -> tuple[str, str] | None:
        """Return the message and the fix of an exterior operation on a state of a degree it does not act on."""
        if name == 'codifferential' and degree == 0:
            return (
                'takes the codifferential of a site state, which has no degree below it',
                'Apply `exterior_derivative` to a site state, or `codifferential` to a state on edges or hyperedges.',
            )
        if name == 'exterior_derivative' and degree >= top:
            return (
                f'takes the exterior derivative of a state of degree {degree}, the highest degree of its complex',
                'Apply `codifferential` to a state of the highest degree, or `exterior_derivative` to a lower one.',
            )
        return None

    @staticmethod
    def _degree_of(value: object, degrees: Mapping[str, int], top: int) -> tuple[set[int], tuple[str, str] | None]:
        """Return the degrees of the values an expression adds up, and the first exterior operation it misapplies."""
        expression = SR(value)
        reached: set[int] = set()
        for reduction in neighborhoods_in(expression):
            problem = PassCompiler._degree_of(reduction.argument, degrees, top)[1]
            if problem is not None:
                return reached, problem
            expression = expression.subs({reduction.call: 0})
        for call in cell_operations_in(expression):
            inner, problem = PassCompiler._degree_of(call.operands()[0], degrees, top)
            if problem is not None:
                return reached, problem
            name = call.operator().name()
            for degree in inner:
                problem = PassCompiler._degree_problem(name, degree, top)
                if problem is not None:
                    return reached, problem
                reached.add(degree + {'exterior_derivative': 1, 'codifferential': -1}.get(name, 0))
            expression = expression.subs({call: 0})
        reached.update(degrees[str(item)] for item in expression.variables() if str(item) in degrees)
        return reached, None

    @staticmethod
    def _sample_issue(record: CensusRecordSpec, name: str) -> IssueSpec | None:
        """Return the issue of a sampling of a hyperedge group, or of a coface reduction outside a cell complex."""
        law = record.law
        assert isinstance(law, SampleLaw)
        if record.degrees.get(str(law.move.target)) == DEGREES['hyperedges']:
            message = f'The {name} samples a state on hyperedges, but a sampling changes a state on sites or edges.'
            fix = 'Sample a group on the sites or edges, and read the hyperedges through `exterior_derivative`.'
            return IssueSpec('law-degree', record.path, message, fix)
        cofaces = [
            item for item in neighborhoods_in(law.objective.expression) if isinstance(item.over, CofaceNeighborhood)
        ]
        if cofaces and not isinstance(record.space, CellComplexSpace):
            lives = type(record.space).__name__ if record.space is not None else 'no space'
            message = f'The {name} reduces over cofaces, which needs a CellComplexSpace, but it lives on {lives}.'
            fix = 'Give the group a `CellComplexSpace`.'
            return IssueSpec('law-space', record.path, message, fix)
        return None

    @staticmethod
    def _degree_issues(records: list[CensusRecordSpec]) -> list[IssueSpec]:
        """Return the exterior operations and samplings on a state of the wrong degree, or outside a cell complex."""
        issues = []
        for record in records:
            name = f'{type(record.law).__name__} of {record.path}'
            include = ('rate',) if isinstance(record.law, JumpLaw) else ('value',)
            values = record.law._get_values(include=include)
            if isinstance(record.law, SampleLaw):
                sampled = PassCompiler._sample_issue(record, name)
                if sampled is not None:
                    issues.append(sampled)
                    continue
                values = [record.law.objective.expression]
            values = [value for value in values if isinstance(value, Expression)]
            if not record.degrees and not any(cell_operations_in(value) for value in values):
                continue
            if not isinstance(record.space, CellComplexSpace):
                lives = type(record.space).__name__ if record.space is not None else 'no space'
                message = f'The {name} applies the exterior calculus, which needs a CellComplexSpace, not {lives}.'
                issues.append(IssueSpec('law-space', record.path, message, 'Give the group a `CellComplexSpace`.'))
                continue
            top = len(record.space.incidences_)
            own = {
                record.degrees.get(str(target))
                for target in record.law._get_targets()
                if isinstance(target, Expression)
            }
            for value in values:
                reached, problem = PassCompiler._degree_of(value, record.degrees, top)
                if problem is not None:
                    issues.append(IssueSpec('law-degree', record.path, f'The {name} {problem[0]}.', problem[1]))
                    break
                if not isinstance(record.law, SampleLaw) and reached - own:
                    message = (
                        f'The {name} changes a state of degree {own.pop()} with a value of degree {sorted(reached)}.'
                    )
                    fix = 'Apply `exterior_derivative` or `codifferential` until the value has the degree of the state.'
                    issues.append(IssueSpec('law-degree', record.path, message, fix))
                    break
        return issues

    @staticmethod
    def _around_issues(records: list[CensusRecordSpec]) -> list[IssueSpec]:
        """Return the reductions over hyperedges of a component outside a network with hyperedges of their size."""
        issues = []
        for record in records:
            name = f'{type(record.law).__name__} of {record.path}'
            include = ('rate',) if isinstance(record.law, JumpLaw) else ('value',)
            values = record.law._get_values(include=include)
            reductions = {str(item.call): item for value in values for item in hyperedges_in(value)}
            space = record.space
            sizes = sorted({len(item) for item in getattr(space, 'hyperedges_', ())})
            for reduction in reductions.values():
                if not isinstance(space, NetworkSpace) or not sizes:
                    lives = type(space).__name__ if space is not None else 'no space'
                    message = (
                        f'The {name} reduces over hyperedges, which needs a network space with hyperedges, but the '
                        f'component lives on {lives}.'
                    )
                    fix = 'Give the group a network space with hyperedges, as `NetworkSpace(graph, hyperedges=...)`.'
                    issues.append(IssueSpec('law-space', record.path, message, fix))
                    break
                if reduction.size is not None and reduction.size not in sizes:
                    message = (
                        f'The {name} reduces over hyperedges of size {reduction.size}, but the hyperedges of its '
                        f'network have the sizes {", ".join(map(str, sizes))}.'
                    )
                    fix = f'Reduce over hyperedges of one of the sizes {", ".join(map(str, sizes))}.'
                    issues.append(IssueSpec('binding', record.path, message, fix))
        return issues

    @staticmethod
    def _field_symbols(nodes: list[NodeSpec]) -> tuple[dict[str, FieldElement], dict[str, float]]:
        """Return the field of every state symbol of a field, and the number of every parameter symbol that has one."""
        fields: dict[str, FieldElement] = {}
        numbers: dict[str, float] = {}
        for node in nodes:
            component = node.component
            label = getattr(component, 'label', '')
            if isinstance(component, FieldElement):
                for name in declarations_of(node.component_type, VariableKind.STATE):
                    fields[f'{name}_{label}'] = component
            values = component.values or {} if isinstance(component, FieldElement) else {}
            for name in declarations_of(node.component_type, VariableKind.PARAMETER):
                value = values.get(name, getattr(component, name, default_of(node.component_type, name)))
                if isinstance(value, int | float) and not isinstance(value, bool):
                    numbers[f'{name}_{label}'] = float(value)
        return fields, numbers

    @staticmethod
    def _is_negative(screening: Expression, fields: Mapping[str, FieldElement], numbers: Mapping[str, float]) -> bool:
        """Check whether a screening reads a state or has a negative square at the numbers of its parameters."""
        if any(str(item) in fields for item in screening.variables()):
            return True
        known = {item: numbers[str(item)] for item in screening.variables() if str(item) in numbers}
        squared = (screening**2).subs(known)
        return not squared.variables() and (not squared.is_real() or float(squared) < 0)

    @staticmethod
    def _operator_issue(
        record: CensusRecordSpec,
        call: Expression,
        fields: Mapping[str, FieldElement],
        numbers: Mapping[str, float],
    ) -> IssueSpec | None:
        """Return the issue of one call of a nonlocal operator in a law of a field, or `None`."""
        name = getattr(call.operator(), 'name', lambda: '')()
        law = PassCompiler._law_name(record)
        operands = call.operands()
        places = {id(fields[str(item)]): fields[str(item)] for item in operands[0].variables() if str(item) in fields}
        if len({id(place.space) for place in places.values()}) > 1:
            message = f'The {law} applies {name} to {operands[0]}, which reads fields of more than one space.'
            return IssueSpec('operator-spaces', record.path, message, 'Apply the operator to the states of one field.')
        if name == 'convolve':
            states = sorted(str(item) for item in operands[1].variables() if str(item) in fields)
            if states:
                message = f'The {law} convolves with a kernel that reads the state {", ".join(states)}.'
                fix = 'Write the kernel in the displacements and the parameters only.'
                return IssueSpec('kernel-state', record.path, message, fix)
        if name == 'inverse_helmholtz' and PassCompiler._is_negative(SR(operands[1]), fields, numbers):
            message = (
                f'The {law} inverts with the screening {operands[1]}, whose square can be negative or that reads a '
                'state.'
            )
            fix = 'Use a screening of parameters whose square is non-negative.'
            return IssueSpec('screening', record.path, message, fix)
        return PassCompiler._place_issue(record, name, next(iter(places.values()))) if places else None

    @staticmethod
    def _place_issue(record: CensusRecordSpec, name: str, place: FieldElement) -> IssueSpec | None:
        """Return the issue of a nonlocal operator that the space or discretization of its field lacks, or `None`."""
        law = PassCompiler._law_name(record)
        space = place.space
        if name in {'convolve', 'shifted'} and not isinstance(space, LatticeSpace):
            verb = 'convolution' if name == 'convolve' else 'shifted read'
            message = (
                f'The {law} uses a {verb}, which needs a lattice space, but the field lives on {type(space).__name__}.'
            )
            return IssueSpec('law-space', record.path, message, 'Give the field a lattice space.')
        periodic = isinstance(space, LatticeSpace) and space.boundary == 'periodic'
        if place.discretization == 'spectral' and name in {'convolve', 'inverse_helmholtz'} and not periodic:
            message = f'The {law} applies the spectral {name} on a lattice without periodic edges.'
            fix = "Give the lattice periodic edges, or use the 'finite-difference' discretization."
            return IssueSpec('law-space', record.path, message, fix)
        return None

    @staticmethod
    def _nonlocal_issues(records: list[CensusRecordSpec], nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the convolutions, shifted reads, and inversions of laws whose space, kernel, or screening is wrong."""
        fields, numbers = PassCompiler._field_symbols(nodes)
        issues = []
        for record in records:
            include = ('rate',) if isinstance(record.law, JumpLaw) else ('value',)
            for value in record.law._get_values(include=include):
                if not FieldLowering._has_operators(value):
                    continue
                calls = [
                    call
                    for name in ('convolve', 'shifted', 'inverse_helmholtz')
                    for call in FieldLowering._operator_calls(SR(value), name)
                ]
                found = [PassCompiler._operator_issue(record, call, fields, numbers) for call in calls]
                issues += [issue for issue in found if issue is not None]
        return issues

    @staticmethod
    def _held_axes(record: CensusRecordSpec, nodes: Mapping[str, NodeSpec]) -> list[str]:
        """Return the class axes of the space of the group of a law and of the fields its interaction reads."""
        spaces = [record.space]
        node = nodes.get(record.path.rpartition('/')[0]) or nodes.get(record.path)
        sources = getattr(node.component, 'sources', ()) if node is not None else ()
        if node is not None and isinstance(sources, list | tuple):
            found = [nodes.get(f'{node.scope}{label}') for label in sources]
            spaces += [getattr(item.component, 'space', None) for item in found if item is not None]
        return [axis.name for space in spaces if isinstance(space, PhaseSpace) for axis in space.axes]

    @staticmethod
    def _collision_issue(record: CensusRecordSpec) -> IssueSpec | None:
        """Return the issue of a collision law on a phase space without two interval class axes, or `None`."""
        space = record.space
        if not (isinstance(record.law, RateLaw) and record.law.cross_section is not None):
            return None
        axes = space.axes if isinstance(space, PhaseSpace) else []
        if len(axes) == _COLLISION_AXES and not any(axis.periodic for axis in axes):
            return None
        lives = type(space).__name__ if space is not None else 'no space'
        message = f'The {PassCompiler._law_name(record)} needs two interval class axes, but the field lives on {lives}.'
        fix = "Declare two interval axes, each an `AxisSpec(name, size, low, high)`, such as 'vx' and 'vy'."
        return IssueSpec('law-space', record.path, message, fix)

    @staticmethod
    def _coordinate_issues(records: list[CensusRecordSpec], nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the coordinates and the reductions over axes of laws whose group is not on a phase space with them."""
        by_path = {node.path: node for node in nodes}
        issues = [issue for record in records if (issue := PassCompiler._collision_issue(record)) is not None]
        for record in records:
            names = {
                str(item).removeprefix(COORDINATE_PREFIX)
                for value in record.law._get_values(include=('value', 'flux', 'rate', 'transition', 'outcomes'))
                if isinstance(value, Expression)
                for item in value.variables()
                if str(item).startswith(COORDINATE_PREFIX)
            }
            held = PassCompiler._held_axes(record, by_path)
            missing = sorted(names - set(held))
            if missing:
                law = PassCompiler._law_name(record)
                lives = type(record.space).__name__ if record.space is not None else 'no space'
                message = (
                    f'The {law} reads the class axes {", ".join(missing)}, but the component lives on {lives}, '
                    f'whose class axes are {held}.'
                )
                fix = f'Give the field a `PhaseSpace` with the axes {", ".join(missing)}, or read one of its axes.'
                issues.append(IssueSpec('law-space', record.path, message, fix))
        return issues

    @staticmethod
    def _space_fix(spaces: tuple[type, ...]) -> str:
        """Return the fix of a neighbourhood that the space does not offer."""
        if ContinuousSpace in spaces:
            return (
                'Give the group a continuous space, such as `ContinuousSpace(extent=(10.0, 10.0))`, or reduce over the '
                'neighbours of its network or lattice with `over=None`, or over a layer with '
                '`AdjacencyNeighborhood(layer)`.'
            )
        return (
            'Give the group a network or lattice space, or reduce over a `RadiusNeighborhood`, a '
            '`NearestNeighborhood`, or a `ConeNeighborhood` in a continuous space, or over a layer of a network with '
            '`AdjacencyNeighborhood(layer)`.'
        )

    @staticmethod
    def _layer_reduction_issue(name: str, path: str, over: object, space: object) -> IssueSpec | None:
        """Return the issue of a reduction over a layer that its network does not have, or `None`."""
        if (
            not isinstance(over, AdjacencyNeighborhood)
            or not isinstance(space, NetworkSpace)
            or over.layer in space.graphs_
        ):
            return None
        known = ', '.join(repr(label) for label in space.graphs_)
        message = (
            f'The {name} reduces over the layer {over.layer!r}, which the network does not have. Its layers are '
            f'{known}.'
        )
        fix = f'Reduce over one of the layers {known}, or add the layer to the network space.'
        return IssueSpec('binding', path, message, fix)

    @staticmethod
    def _neighborhood_issues(records: list[CensusRecordSpec], nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the neighbourhood reductions whose space does not offer them, and the cones without a direction."""
        by_path = {node.path: node for node in nodes}
        issues = []
        for record in records:
            name = f'{type(record.law).__name__} of {record.path}'
            include = ('rate',) if isinstance(record.law, JumpLaw) else ('value',)
            values = record.law._get_values(include=include)
            reductions = {str(item.call): item for value in values for item in PassCompiler._reductions_of(value)}
            for reduction in reductions.values():
                over = reduction.over
                if over is not None and not isinstance(over, BaseNeighborhood):
                    error_msg = f'The {name} reduces over {over!r}, which is not a neighbourhood.'
                    raise ValidationError(error_msg)
                spaces = _OWN_NEIGHBORHOOD if over is None else NeighborhoodKernel._spaces_of(over)
                described = 'the neighbours of its space' if over is None else repr(over)
                if not isinstance(record.space, spaces):
                    lives = type(record.space).__name__ if record.space is not None else 'no space'
                    message = (
                        f'The {name} reduces over {described}, which needs a {PassCompiler._names(spaces)}, but the '
                        f'component lives on {lives}.'
                    )
                    issues.append(IssueSpec('law-space', record.path, message, PassCompiler._space_fix(spaces)))
                    continue
                layered = PassCompiler._layer_reduction_issue(name, record.path, over, record.space)
                if layered is not None:
                    issues.append(layered)
                    continue
                node = by_path.get(record.owner)
                if node is None or not isinstance(record.space, ContinuousSpace) or over is None:
                    continue
                states = declarations_of(node.component_type, VariableKind.STATE)
                if NeighborhoodKernel._position_state(states) is None:
                    message = (
                        f'The {name} reduces over {over!r}, which measures distances between positions, but '
                        f'{record.owner} declares no vector position state.'
                    )
                    fix = "Declare the position with `StateSpec(vector=True, role='position')`."
                    issues.append(IssueSpec('law-domain', record.path, message, fix))
                    continue
                if isinstance(over, ConeNeighborhood):
                    try:
                        NeighborhoodKernel._cone_direction(over, states, getattr(record.space, 'n_dim_', 0))
                    except ValidationError as error:
                        fix = (
                            "Declare the position of order 2, `StateSpec(vector=True, order=2, role='position')`, or "
                            "name a direction state, such as `ConeNeighborhood(r, angle, direction='theta')`."
                        )
                        issues.append(IssueSpec('law-domain', record.path, f'The {name}: {error}', fix))
        return issues

    @staticmethod
    def _solver_issues(solver: object, census: CensusSpec, system: System) -> list[IssueSpec]:
        """Return the arguments the solver passed rejects and the requirements of a system it does not support."""
        records = census.records
        stepper = type(solver)
        needs: dict[str, Iterable[object]] = {
            'time': {kind for kind in FractionalFormPass._declared_times(system).values() if kind is not None},
            'laws': {type(record.law) for record in records},
            'space': {None if record.space is None else type(record.space) for record in records},
        }
        issues = []
        try:
            if isinstance(solver, BaseConfig):
                solver.check()
        except SkcomplexError as error:
            fix = f'Correct the arguments of {stepper.__name__} as the message describes.'
            issues.append(IssueSpec('solver-argument', system.label, str(error), fix, error=type(error)))
        support = BaseCompiler._support_of(stepper)
        for category, terms in needs.items():
            missing = support.missing(category, terms)
            if missing:
                message = f'The solver {stepper.__name__} does not support the {category} {", ".join(missing)}.'
                fix = f'Pass a solver that supports the {category} {", ".join(missing)}.'
                issues.append(IssueSpec('solver-unsupported', system.label, message, fix, error=SolverError))
        return issues

    @staticmethod
    def _matrix_problem(matrix: np.ndarray) -> str | None:
        """Return why a square covariance matrix is not symmetric positive semidefinite, or `None` when it is."""
        asymmetry = np.abs(matrix - matrix.T)
        if np.max(asymmetry) > _COVARIANCE_TOLERANCE * max(1.0, float(np.max(np.abs(matrix)))):
            row, column = np.unravel_index(int(np.argmax(asymmetry)), matrix.shape)
            return f'Its covariance is not symmetric: entry ({row}, {column}) is {matrix[row, column]}.'
        values = np.linalg.eigvalsh(matrix)
        if values[0] < -_COVARIANCE_TOLERANCE * max(1.0, float(np.max(np.abs(values)))):
            return f'Its covariance is not positive semidefinite: it has the negative eigenvalue {values[0]:.6g}.'
        return None

    @staticmethod
    def _covariance_problem(wiener: WienerProcessSpec) -> str | None:
        """Return why the dimension or the covariance of a `WienerProcessSpec` is invalid, or `None` when valid."""
        dimension = wiener.dimension
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            return f'Its dimension {dimension!r} is not a positive integer.'
        if wiener.covariance is None:
            return None
        try:
            matrix = np.asarray(wiener.covariance, dtype=float)
        except (TypeError, ValueError):
            return f'Its covariance {wiener.covariance!r} is not a matrix of numbers.'
        if matrix.shape != (dimension, dimension):
            return f'Its covariance has the shape {matrix.shape}, not ({dimension}, {dimension}).'
        return PassCompiler._matrix_problem(matrix)

    @staticmethod
    def _wiener_issues(system: System, nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the Wiener processes outside continuous time or with an invalid covariance."""
        times = FractionalFormPass._declared_times(system)
        issues = []
        for path, wiener in LayoutPass._noise_processes(nodes).items():
            time = times.get(path.rpartition('/')[0])
            if time is not None and not issubclass(time, ContinuousTime):
                message = f'The Wiener processes {path} exist only in continuous time.'
                issues.append(IssueSpec('wiener-time', path, message, 'Declare continuous time for its system.'))
            problem = PassCompiler._covariance_problem(wiener)
            if problem is not None:
                message = f'The Wiener processes {path} are invalid. {problem}'
                fix = 'Give a symmetric positive semidefinite covariance with one row and column per process.'
                issues.append(IssueSpec('wiener-covariance', path, message, fix))
        return issues

    @staticmethod
    def _law_name(record: CensusRecordSpec) -> str:
        """Return the name of the law of a census record."""
        return f'{type(record.law).__name__} of {record.path}'

    @staticmethod
    def _process_name(key: ProcessKey) -> str:
        """Return the name of a process key."""
        kind, path, item = key
        return (
            f'the local process {item!r} of {path}'
            if kind == 'local'
            else f'process {item} of the Wiener processes {path}'
        )

    @staticmethod
    def _noise_records(records: list[CensusRecordSpec]) -> list[CensusRecordSpec]:
        """Return the records of noise laws."""
        return [record for record in records if isinstance(record.law, NoiseLaw)]

    @staticmethod
    def _handle_issues(records: list[CensusRecordSpec]) -> list[IssueSpec]:
        """Return the laws that use the handle of a Wiener process outside the process of a noise law."""
        issues = []
        for record in records:
            if record.handles:
                message = (
                    f'The {PassCompiler._law_name(record)} uses the Wiener handle {", ".join(record.handles)} in a '
                    'value, but a handle is only the process of a noise law.'
                )
                fix = 'Name the handle as the process of a noise law, `NoiseLaw(target, value, process=s.<role>.dW)`.'
                issues.append(IssueSpec('wiener-handle', record.path, message, fix))
        return issues

    @staticmethod
    def _convention_issues(records: list[CensusRecordSpec]) -> list[IssueSpec]:
        """Return the processes that noise laws of different conventions share."""
        sharing: dict[ProcessKey, list[CensusRecordSpec]] = {}
        for record in PassCompiler._noise_records(records):
            for key in dict.fromkeys(record.processes):
                if key is not None:
                    sharing.setdefault(key, []).append(record)
        issues = []
        for process, sharers in sharing.items():
            conventions = {record.law.convention for record in sharers if isinstance(record.law, NoiseLaw)}
            if len(conventions) > 1:
                laws = ' and '.join(
                    f'the {item.law.convention} {PassCompiler._law_name(item)}'
                    for item in sharers
                    if isinstance(item.law, NoiseLaw)
                )
                shared = PassCompiler._process_name(process)
                message = f'{laws[0].upper()}{laws[1:]} share {shared} with different conventions.'
                fix = 'Declare one convention for every noise law of the process.'
                issues.append(IssueSpec('noise-convention-mixed', sharers[0].path, message, fix))
        return issues

    @staticmethod
    def _sharer(
        record: CensusRecordSpec,
        noisy: list[CensusRecordSpec],
        wieners: dict[str, WienerProcessSpec],
    ) -> CensusRecordSpec | None:
        """Return a noise law of another component that drives a state a law's amplitude reads through its process."""
        for key, reads in zip(record.processes, record.reads, strict=True):
            if key is None:
                continue
            for item in noisy:
                if item.path == record.path:
                    continue
                for state, process in zip(item.driven, item.processes, strict=True):
                    if process is not None and state in reads and NoisePass._process_covariance(key, process, wieners):
                        return item
        return None

    @staticmethod
    def _scope_issues(records: list[CensusRecordSpec], wieners: dict[str, WienerProcessSpec]) -> list[IssueSpec]:
        """Return the noise laws of a convention whose drift correction needs an amplitude of another component."""
        noisy = PassCompiler._noise_records(records)
        issues = []
        for record in noisy:
            if not isinstance(record.law, NoiseLaw) or record.law.convention == 'ito':
                continue
            other = PassCompiler._sharer(record, noisy, wieners)
            if other is not None:
                message = (
                    f'The {PassCompiler._law_name(record)} depends on a state that the {PassCompiler._law_name(other)} '
                    'drives through a shared process, so its drift correction needs an amplitude outside its component.'
                )
                fix = 'Use the Itô convention, or write both laws in one component.'
                issues.append(IssueSpec('noise-correction-scope', record.path, message, fix))
        return issues

    @staticmethod
    def _noise_needs(
        records: list[CensusRecordSpec],
        wieners: dict[str, WienerProcessSpec],
    ) -> dict[NoiseKind, CensusRecordSpec]:
        """Return the noise structures a system uses, each with the first law that needs it."""
        needs: dict[NoiseKind, CensusRecordSpec] = {}
        states: dict[object, set[tuple[str, int]]] = {}
        first: dict[object, CensusRecordSpec] = {}
        spread: set[object] = set()
        for record in PassCompiler._noise_records(records):
            if isinstance(record.law, NoiseLaw) and record.law.conserved:
                needs.setdefault(NoiseKind.CONSERVED, record)
                continue
            for key, state in zip(record.processes, record.driven, strict=True):
                if key is None:
                    needs.setdefault(NoiseKind.DIAGONAL, record)
                    continue
                states.setdefault(key, set()).add(state)
                first.setdefault(key, record)
                if key[0] == 'wiener' and record.members > 1:
                    spread.add(key)
                if key[0] == 'wiener' and wieners[key[1]].covariance is not None:
                    needs.setdefault(NoiseKind.CORRELATED, record)
        for process, driven in states.items():
            shared = len(driven) > 1 or process in spread
            needs.setdefault(NoiseKind.SHARED if shared else NoiseKind.DIAGONAL, first[process])
        return needs

    @staticmethod
    def _noise_runners(solver: object, backend: object) -> list[tuple[str, type]]:
        """Return the classes of the solver and the backend passed that run the noise of a system."""
        runners: list[tuple[str, type]] = []
        if solver is not None:
            runners.append(('solver', type(solver)))
        if backend is not None:
            runners.append(('backend', type(backend)))
        return runners

    @staticmethod
    def _noise_fix(term: NoiseKind) -> str:
        """Return the fix of a noise structure that the solver or the backend passed does not support."""
        return f'Pass a solver and a backend that support {term} noise.'

    @staticmethod
    def _noise_support_issues(
        records: list[CensusRecordSpec],
        wieners: dict[str, WienerProcessSpec],
        solver: object,
        backend: object,
    ) -> list[IssueSpec]:
        """Return the noise structures of a system that the chosen solver or backend does not support."""
        needs = PassCompiler._noise_needs(records, wieners)
        if not needs:
            return []
        issues = []
        for role, runner in PassCompiler._noise_runners(solver, backend):
            supported = BaseCompiler._support_of(runner).noise
            for term, record in needs.items():
                if term not in supported:
                    name = PassCompiler._law_name(record)
                    message = f'The {name} needs {term} noise, which the {role} {runner.__name__} lacks.'
                    fix = PassCompiler._noise_fix(term)
                    issues.append(IssueSpec('noise-unsupported', record.path, message, fix, error=SolverError))
        return issues

    @staticmethod
    def _noise_issues(compilation: Compilation, records: list[CensusRecordSpec]) -> list[IssueSpec]:
        """Return the problems of the Wiener processes and the noise laws of a system."""
        nodes = Compilation._required(compilation.nodes, 'nodes')
        solver, backend = compilation.solver, compilation.backend
        wieners = LayoutPass._noise_processes(nodes)
        issues = PassCompiler._wiener_issues(compilation.system, nodes) + PassCompiler._handle_issues(records)
        if any(issue.code == 'wiener-covariance' for issue in issues):
            return issues
        issues += PassCompiler._convention_issues(records)
        issues += PassCompiler._scope_issues(records, wieners)
        return issues + PassCompiler._noise_support_issues(records, wieners, solver, backend)

    @staticmethod
    def _law_values(record: CensusRecordSpec) -> Iterator[object]:
        """Yield the symbolic values of the law of a census record, with the items of vectors."""
        for value in record.law._get_values():
            if isinstance(value, FreeModuleElement):
                yield from value
            elif isinstance(value, Expression):
                yield value

    @staticmethod
    def _held_reductions(census: CensusSpec, time: Expression) -> list[_HeldReductionSpec]:
        """Return the reductions of agents that the laws of a census read, each once, in the order they are read."""
        found: dict[tuple[str, ...], _HeldReductionSpec] = {}
        for record in census.records:
            for value in PassCompiler._law_values(record):
                for reduction in reductions_in(value):
                    held = ReductionLowering._held_reduction(reduction, census.agents, time, record.path)
                    if held is not None:
                        found.setdefault(held.key, held)
        return list(found.values())

    @staticmethod
    def _census_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the issues of the components that fail to declare or collect, unless their binding is the problem."""
        census = Compilation._required(compilation.census, 'census')
        nodes = Compilation._required(compilation.nodes, 'nodes')
        bindings = PassCompiler._binding_issues(nodes)
        shapes = PassCompiler._shape_issues(nodes)
        unbound = {issue.path for issue in bindings} | {issue.path.rpartition('.')[0] for issue in shapes}
        issues = []
        for path, failure in census.errors:
            if path in unbound:
                continue
            code, fix = _FIXES.get(type(failure), ('value', 'Correct the component as the message describes.'))
            issues.append(IssueSpec(code, path, str(failure), fix, error=type(failure)))
        return issues + bindings + shapes

    @staticmethod
    def _probability_sum(total: object) -> float | None:
        """Return the sum of the outcome probabilities of one encounter as a number, or `None` for a symbol."""
        value = SR(total).simplify_full() if isinstance(total, Expression) else SR(total)
        return float(value) if value.is_numeric() else None

    @staticmethod
    def _unnormalized(law: EncounterLaw | BilinearLaw) -> list[tuple[int, int, float]]:
        """Return the candidate and field activities of the encounters whose outcome probabilities do not sum to one."""
        value = law.transition if isinstance(law, EncounterLaw) else law.outcomes
        if value is None:
            return []
        sums = CompileEncountersPass._kinetic_table(value).sum(axis=2)
        rates = np.broadcast_to(CompileEncountersPass._kinetic_table(law.rate), sums.shape)
        found = []
        for (candidate, field_activity), total in np.ndenumerate(sums):
            rate = rates[candidate, field_activity]
            if SR(rate).is_numeric() and float(SR(rate)) == 0:
                continue
            value = PassCompiler._probability_sum(total)
            if value is not None and abs(value - 1) > _PROBABILITY_TOLERANCE:
                found.append((candidate, field_activity, value))
        return found

    @staticmethod
    def _probability_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the encounter laws whose outcome probabilities do not sum to one over the outcomes of an encounter.

        Args:
            compilation:
                The compilation state after planning.

        Returns:
            issues:
                The issues.
        """
        issues = []
        for record in Compilation._required(compilation.census, 'census').records:
            law = record.law
            if not isinstance(law, EncounterLaw | BilinearLaw):
                continue
            found = PassCompiler._unnormalized(law)
            if not found:
                continue
            listed = '; '.join(
                f'candidate activity {candidate} with field activity {field_activity} sums to {value:.6g}'
                for candidate, field_activity, value in found[:_LISTED_ENCOUNTERS]
            )
            more = f', and {len(found) - _LISTED_ENCOUNTERS} more' if len(found) > _LISTED_ENCOUNTERS else ''
            message = (
                f'The {type(law).__name__} of {record.path} has outcome probabilities that do not sum to one over the '
                f'outcomes of an encounter: {listed}{more}.'
            )
            fix = 'Give outcome probabilities that sum to one over the outcomes of every encounter with a nonzero rate.'
            issues.append(IssueSpec('encounter-probabilities', record.path, message, fix))
        return issues

    @staticmethod
    def _fixed_groups(
        request: TopologyRequestSpec,
        owner: NodeSpec | None,
        groups: list[NodeSpec],
    ) -> tuple[list[str], bool]:
        """Return the fixed groups a request changes, and whether its layer lacks an edge group."""
        edges = [node for node in groups if node.component.on == 'edges' and node.component.layer == request.layer]
        sites = [node for node in groups if node.component.on == 'sites']
        changed = []
        if TopologyKernel._changes_edges(request):
            changed += edges
        if request.kind in {'add-node', 'remove-node'}:
            changed += sites
        if owner is not None:
            changed.append(owner)
        fixed = list(dict.fromkeys(node.path for node in changed if not node.component.variable))
        return fixed, TopologyKernel._changes_edges(request) and not edges

    @staticmethod
    def _fixed_group_issues(records: list[CensusRecordSpec], nodes: list[NodeSpec]) -> list[IssueSpec]:
        """Return the topology requests that change fixed groups, or edges of a layer without a variable edge group."""
        by_path = {node.path: node for node in nodes}
        issues = []
        for record in records:
            try:
                request = TopologyKernel._request_of(record.law)
            except ValidationError as error:
                issues.append(
                    IssueSpec('value', record.path, str(error), 'Fire one request per jump, with increment 1.'),
                )
                continue
            if request is None:
                continue
            space = record.space
            if not isinstance(space, NetworkSpace):
                spaces = {id(node.space): node.space for node in nodes if isinstance(node.space, NetworkSpace)}
                space = next(iter(spaces.values())) if len(spaces) == 1 else space
            groups = [node for node in nodes if isinstance(node.component, GroupElement) and node.space is space]
            owner = by_path.get(record.owner)
            owner = owner if owner is not None and isinstance(owner.component, GroupElement) else None
            fixed, missing = PassCompiler._fixed_groups(request, owner, [node for node in groups if node.component.on])
            if not fixed and not missing:
                continue
            listed = ', '.join(fixed) or 'none'
            where = f'the layer {request.layer!r}' if request.layer is not None else 'the network'
            message = (
                f'The {type(record.law).__name__} of {record.path} requests {request!r}, which changes the fixed '
                f'groups {listed}.'
            )
            if missing:
                message += f' {where[0].upper()}{where[1:]} has no group on its edges.'
            fix = (
                'Give the groups on the nodes and the edges of the network `variable=True`, with a group on the edges '
                'of every layer the request changes, `GroupElement(..., on="edges", variable=True)`.'
            )
            issues.append(IssueSpec('group-fixed', record.path, message, fix))
        return issues

    @staticmethod
    def _request_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the topology requests of a compilation that change fixed groups or a layer without an edge group."""
        records = Compilation._required(compilation.census, 'census').records
        return PassCompiler._fixed_group_issues(records, Compilation._required(compilation.nodes, 'nodes'))
