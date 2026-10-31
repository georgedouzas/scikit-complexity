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
from sklearn.base import clone

from ...base import BaseConfig
from ...core.elements import GroupElement
from ...core.interactions import HyperedgeInteraction, NeighborInteraction
from ...core.laws import EncounterLaw, JumpLaw, NoiseLaw, ProliferationLaw
from ...core.laws.utils import neighborhoods_in, reductions_in
from ...core.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    CayleySpace,
    ConeNeighborhood,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    StaticTime,
)
from ...core.specs import CensusRecordSpec, CensusSpec, NodeSpec
from ...core.system import System
from ...errors import (
    ParameterError,
    PipelineError,
    SkcomplexError,
    SolverError,
    SpaceError,
    SpaceTooLargeError,
    ValidationError,
)
from ...specs import (
    IssueSpec,
    NeighborReductionSpec,
    ObservableSpec,
    TopologyRequestSpec,
    WienerProcessSpec,
)
from ...specs._held_reduction_spec import _HeldReductionSpec
from ...specs._issue_spec import _ERROR
from ...types import DomainKind, FormKind, NoiseKind, ProcessKey, VariableKind
from .._utils import declarations_of, symbol
from ..backends import BaseBackend
from ..problems import BaseProblemSpec, PartitionSpec, ProblemSpec
from ..problems._neighborhood_kernel import NeighborhoodKernel
from ..problems._topology_kernel import TopologyKernel
from ._base import _FIXES, BaseCompiler, BasePass
from ._compilation import Compilation
from ._observables import _check_held_reads, _held_observables, _with_held_observables, _with_held_reductions
from .passes._compile_encounters import CompileEncountersPass
from .passes._conserved_noise import ConservedNoisePass
from .passes._core import _PASSES
from .passes._fractional_form import FractionalFormPass
from .passes._lattices import LatticeProblemPass
from .passes._layout import LayoutPass
from .passes._noise import NoisePass
from .passes._quantum_compiler import QuantumCompiler
from .passes._reduction_lowering import ReductionLowering

_COVARIANCE_TOLERANCE = 1e-12
_PROBABILITY_TOLERANCE = 1e-9
_LISTED_ENCOUNTERS = 3
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
            The compilation planned so far, or `None` when the system tree cannot be flattened.

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
        self.compilation_, issues = self._counted(system)
        if self.compilation_ is None or self.compilation_.census is None:
            return issues
        issues = self._issues_of(self.compilation_)
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
        )

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

        The compilation stops after flattening for a system that another compiler compiles. When laws read
        reductions of agents, the compilation starts again from a system tree that holds each of them in an element.
        """
        compilation = self._new_compilation(system)
        try:
            self._run_passes(compilation, until='nodes')
        except SkcomplexError as error:
            fix = 'Pass elements, groups, interactions, and systems as children.'
            return None, [self._issue_of(error, system.label, 'structure', fix)]
        if QuantumCompiler._contains_quantum(system):
            return compilation, [self._compiler_issue(system, 'QuantumCompiler')]
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
        issues += PassCompiler._neighborhood_issues(records, nodes)
        issues += PassCompiler._space_issues(system, nodes)
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
            bound = [
                by_path.get(f'{node.scope}{getattr(interaction, name)}')
                for name in roles
                if getattr(interaction, name, None) is not None
            ]
            members = [item for item in bound if item is not None and getattr(item.component, 'on', None) != 'edges']
            target = next(iter(members), None)
            space = target.space if target is not None else node.space
            offered = (
                isinstance(space, LatticeSpace | NetworkSpace | CayleySpace)
                if over == 'neighbors'
                else bool(
                    getattr(space, 'hyperedges', ()),
                )
            )
            if not offered and over == 'neighbors' and len(members) == len(bound):
                continue
            if not offered:
                needs = 'a lattice or network space' if over == 'neighbors' else 'a network space with hyperedges'
                message = f'The binding over {over} needs {needs}, but the bound members live on {space!r}.'
                issues.append(IssueSpec('binding', node.path, message, f'Give the bound group {needs}.'))
                continue
            issues += PassCompiler._layer_issues(node, space)
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
        """Return the neighbourhood reductions whose space does not offer them, and the cones without a heading."""
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
                    fix = 'Declare the position with `StateSpec(vector=True, position=True)`.'
                    issues.append(IssueSpec('law-domain', record.path, message, fix))
                    continue
                if isinstance(over, ConeNeighborhood):
                    try:
                        NeighborhoodKernel._cone_heading(over, states, getattr(record.space, 'n_dim_', 0))
                    except ValidationError as error:
                        fix = (
                            'Declare the position of order 2, `StateSpec(vector=True, position=True, order=2)`, or '
                            "name a heading state, such as `ConeNeighborhood(r, angle, heading='theta')`."
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
        bindings = PassCompiler._binding_issues(Compilation._required(compilation.nodes, 'nodes'))
        unbound = {issue.path for issue in bindings}
        issues = []
        for path, failure in census.errors:
            if path in unbound:
                continue
            code, fix = _FIXES.get(type(failure), ('value', 'Correct the component as the message describes.'))
            issues.append(IssueSpec(code, path, str(failure), fix, error=type(failure)))
        return issues + bindings

    @staticmethod
    def _probability_sum(total: object) -> float | None:
        """Return the sum of the outcome probabilities of one encounter as a number, or `None` for a symbol."""
        value = SR(total).simplify_full() if isinstance(total, Expression) else SR(total)
        return float(value) if value.is_numeric() else None

    @staticmethod
    def _unnormalized(law: EncounterLaw | ProliferationLaw) -> list[tuple[int, int, float]]:
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
            if not isinstance(law, EncounterLaw | ProliferationLaw):
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
