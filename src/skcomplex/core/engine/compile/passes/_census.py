"""Take the census of the laws every component of a system declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix
from sage.symbolic.function import SymbolicFunction

from ....components.base import BaseComponent
from ....components.elements import FieldElement, GroupElement
from ....components.interactions import BaseInteraction
from ....components.laws import (
    BaseLaw,
    BoundaryLaw,
    CallSpec,
    ConstraintLaw,
    EventLaw,
    JumpLaw,
    LinearObjectiveSpec,
    LinearProgramLaw,
    LocalRuleLaw,
    MoveSpec,
    NoiseLaw,
    ObjectiveLaw,
    ObjectiveSpec,
    RateLaw,
    RuleLaw,
    SampleLaw,
    SamplerSpec,
    TransferLaw,
    UpdateLaw,
)
from ....components.laws._flow_term import FlowTermLaw
from ....components.laws.utils import laplacian, shifted
from ....components.laws.utils._history import derivative
from ....components.laws.utils._operators import _OperatorFunction
from ....components.laws.utils._reductions import hyperedges_in, neighborhoods_in, reductions_in
from ....components.laws.utils._relaxations import relaxation_of
from ....components.laws.utils._tensors import distribute_operators
from ....components.spaces import CellComplexSpace, LatticeSpace, RadiusNeighborhood
from ....components.systems import CompositeSystem
from ....enums import ElementKind, FormKind, VariableKind
from ....errors import RoleElementError, SkcomplexError, TensorShapeError, TensorSymmetryError, ValidationError
from ....specs import EntrySpec, TensorSpec
from ....types import ProcessKey
from ..._utils import components_of, declarations_of, symbol, symbols_of, vector
from ...enums import ReadKind
from ...problems._bilinear_law import BilinearLaw
from ...problems._block_spec import BlockSpec
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._edge_search import EdgeSearch
from ...problems._kernel_spec import KernelSpec
from ...problems._node_spec import _NodeSpec
from ...problems._pair_search import PairSearch
from ...problems._problem_spec import ProblemSpec
from ...problems._topology_kernel import TopologyKernel
from .._array_update_law import ArrayUpdateLaw
from .._base import BasePass
from .._census_record_spec import _CensusRecordSpec
from .._census_spec import _CensusSpec
from .._compilation import Compilation
from .._field_lowering import FieldLowering
from .._observables import ObservableEvaluator
from .._observer import Observer
from .._relaxation_law import RelaxationLaw
from .._renewal_law import RenewalLaw
from ._cells import DEGREES
from ._hyperedges import HYPEREDGE_LAWS, HyperedgeTables
from ._kernels import KernelsPass
from ._layout import LayoutPass
from ._noise import NoisePass
from ._rows import RowsPass
from ._specs import SpecsPass

EDGE_REQUESTS = ('remove-edge', 'rewire')
SHIFT = 'shift_member'
SYMMETRY_TOLERANCE = 1e-9
SYMMETRY_TRIALS = 3


class CensusPass(BasePass):
    """A pass that takes the census of the laws of the system.

    It keeps going past failing components.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('nodes',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('census',)

    def _run(self, inputs: object = None) -> None:
        """Take the census of the laws of the nodes."""
        compilation = self.compilation_
        compilation.census = self._get_census_of(
            compilation.system,
            Compilation._get_required(compilation.nodes, 'nodes'),
            frozenset(compilation.trained),
            compilation.solver,
        )

    @staticmethod
    def _get_state_reads(spec: KernelSpec, value: object) -> frozenset[tuple[str, int]]:
        """Return the state components that a law value depends on."""
        if not isinstance(value, Expression):
            return frozenset()
        return frozenset(
            (reference.entry.name, reference.column)
            for item in value.variables()
            if (reference := spec.references.get(item)) is not None and reference.is_state
        )

    @staticmethod
    def _get_noise_census(
        spec: KernelSpec,
        law: object,
        declarations: DeclarationsSpec,
    ) -> tuple[tuple[ProcessKey | None, ...], tuple[tuple[str, int], ...], tuple[frozenset[tuple[str, int]], ...]]:
        """Return the processes, driven states, and read states of a noise law.

        They are given for every target component of the law.
        """
        if not isinstance(law, NoiseLaw):
            return (), (), ()
        targets = components_of(law._get_targets()[0])
        values = (
            components_of(law.value)
            if isinstance(law.value, FreeModuleElement | list | tuple)
            else [law.value] * len(targets)
        )
        references = [spec.references.get(item) for item in targets]
        return (
            tuple(NoisePass._get_process_keys(spec, law, declarations)),
            tuple((item.entry.name, item.column) for item in references if item is not None),
            tuple(CensusPass._get_state_reads(spec, value) for value in values),
        )

    @staticmethod
    def _is_vector(entry: EntrySpec, elements: dict[str, BlockSpec]) -> bool:
        """Check whether the state of a layout entry is a vector of the space dimension."""
        path, _, name = entry.name.rpartition('.')
        block = elements.get(path)
        if block is None:
            return False
        declaration = declarations_of(block.node.component_type, VariableKind.STATE).get(name)
        return declaration is not None and (declaration.vector or declaration.tensor is not None)

    @staticmethod
    def _handle_uses(law: BaseLaw, handles: Mapping[Expression, tuple[str, int]]) -> tuple[str, ...]:
        """Return the Wiener handles that a law uses outside the process of a noise law."""
        if not handles:
            return ()
        symbols = symbols_of(
            [value for value in law._get_values(exclude=('process',)) if isinstance(value, Expression)],
        )
        return tuple(sorted(str(item) for item in symbols if item in handles))

    @staticmethod
    def _get_component_kind(node: _NodeSpec) -> ElementKind:
        """Return the kind of the component of a node."""
        if isinstance(node.component, GroupElement) and node.component.variable:
            return ElementKind.VARIABLE_GROUP
        if isinstance(node.component, GroupElement):
            parts = {'sites': ElementKind.SITE_GROUP, 'edges': ElementKind.EDGE_GROUP}
            parts['hyperedges'] = ElementKind.HYPEREDGE_GROUP
            return parts.get(node.component.on or '', ElementKind.GROUP)
        return ElementKind.ELEMENT

    @staticmethod
    def _get_equation_targets(spec: KernelSpec, law: BaseLaw) -> tuple[EntrySpec, ...]:
        """Return the entry of the state an equation names as its target.

        It is none for any other law or target.
        """
        target = law.target if isinstance(law, ConstraintLaw) else None
        reference = spec.references.get(target) if isinstance(target, Expression) else None
        return (reference.entry,) if reference is not None and reference.is_state else ()

    @staticmethod
    def _get_cell_degrees(spec: KernelSpec, declarations: DeclarationsSpec) -> dict[str, int]:
        """Return the cell degree of every symbol of a spec reading a cell complex group.

        The spec reads the group by member. A law that runs once per edge reads both ends and the edge, so it holds no
        degree of its own.
        """
        if spec.pairs is not None:
            return {}
        degrees = {}
        for item, reference in spec.references.items():
            block = declarations.blocks.get(reference.entry.name.rpartition('.')[0])
            part = getattr(block.node.component, 'on', None) if block is not None else None
            if block is not None and isinstance(block.node.space, CellComplexSpace) and part in DEGREES:
                degrees[str(item)] = DEGREES[part]
        return degrees

    @staticmethod
    def _get_records_of(
        block: BlockSpec,
        elements: dict[str, BlockSpec],
        observer: Observer,
        declarations: DeclarationsSpec,
        agents: set[str],
        *,
        solver: object = None,
        scope: str = '',
    ) -> list[_CensusRecordSpec]:
        """Return the census records of the laws of one component.

        A linear program has as many members as the rows its laws settle, such as the members of two groups it reads
        row by row. The moves of the solver name states relative to `scope`, the path of the system that holds it.
        """
        records = []
        specs, _ = SpecsPass._get_specs_of(block, elements, declarations, observer)
        for spec in specs:
            component = CensusPass._get_component_of(block.node)
            laws = CensusPass._get_declared_laws(spec, component, solver, scope)
            programs = any(isinstance(law, LinearProgramLaw) for law in laws)
            rows = RowsPass._get_settled(spec, list(laws)).rows if programs else spec.rows
            for law in laws:
                items = [item for target in law._get_targets() for item in ProblemSpec._get_real_components_of(target)]
                references = [spec.references.get(item) for item in items if isinstance(item, Expression)]
                targets = tuple(dict.fromkeys(item.entry for item in references if item is not None))
                targets = targets or CensusPass._get_equation_targets(spec, law)
                owner = CensusPass._get_law_owner(spec, law, agents)
                vectors = frozenset(entry.name for entry in targets if CensusPass._is_vector(entry, elements))
                kind = CensusPass._get_component_kind(elements.get(owner, block).node)
                processes, driven, reads = CensusPass._get_noise_census(spec, law, declarations)
                records.append(
                    _CensusRecordSpec(
                        block.node.path,
                        owner,
                        law,
                        targets,
                        vectors,
                        kind,
                        SpecsPass._get_group_space(elements.get(owner, block)),
                        members=rows if isinstance(law, LinearProgramLaw) else spec.rows,
                        handles=CensusPass._handle_uses(law, declarations.handles),
                        degrees=CensusPass._get_cell_degrees(spec, declarations),
                        processes=processes,
                        driven=driven,
                        reads=reads,
                        differentiated=CensusPass._get_differentiated(spec, law),
                    ),
                )
        return records

    @staticmethod
    def _get_differentiated(spec: KernelSpec, law: BaseLaw) -> frozenset[str]:
        """Return the elements and groups whose states an equation differentiates.

        It is empty for any other law.
        """
        if not isinstance(law, ConstraintLaw):
            return frozenset()
        found = set()
        for value in law._get_values():
            if not isinstance(value, Expression):
                continue
            for call in value.find(derivative(SR.wild(0), SR.wild(1))):
                reference = spec.references.get(call.operands()[0])
                if reference is not None and reference.is_state:
                    found.add(reference.entry.name.rpartition('.')[0])
        return frozenset(found)

    @staticmethod
    def _get_census_of(
        system: CompositeSystem,
        nodes: list[_NodeSpec],
        trained: frozenset[str] = frozenset(),
        solver: object = None,
    ) -> _CensusSpec:
        """Return the census of the laws of a system.

        The errors of the components that fail are included. The census declares every component and collects its laws
        as compilation does, without lowering them, and keeps going past a failing component, so validation reports
        every problem at once.

        Args:
            system:
                The system, with the observables of the run.

            nodes:
                The nodes of the system.

            trained:
                The dotted paths of the parameters a training trains, which are adaptive.

            solver:
                The solver of the compilation, whose moves turn objectives of members into sweeps, or `None`.

        Returns:
            census:
                The census.
        """
        declarations = DeclarationsSpec(time=symbol('t'))
        errors: list[tuple[str, SkcomplexError]] = []
        blocks = []
        groups = LayoutPass._get_group_sizes(nodes)
        for node in nodes:
            try:
                declarations, block = LayoutPass._declare_component(declarations, node, groups, trained)
            except SkcomplexError as error:
                errors.append((node.path, error))
                continue
            blocks.append(block)
        elements = {block.node.path: block for block in blocks if not block.node.interaction}
        elements.update(declarations.processes)
        live = ObservableEvaluator._get_lowerable_observables(system, nodes)
        observer = Observer(declarations, live).prepare()
        agents = {block.node.path for block in blocks if block.node.is_agent}
        solvers = CensusPass._get_solvers(system, (solver, ''))
        records = []
        for block in blocks:
            nested, scope = solvers.get(block.node.scope, (solver, ''))
            try:
                records += CensusPass._get_records_of(
                    block,
                    elements,
                    observer,
                    declarations,
                    agents,
                    solver=nested,
                    scope=scope,
                )
            except SkcomplexError as error:
                errors.append((block.node.path, error))
        flowing = CensusPass._get_flowing(records)
        records = [
            replace(record, law=CensusPass._get_corrected(record.law, record.owner, flowing)) for record in records
        ]
        return _CensusSpec(system, records, errors, declarations.agents)

    @staticmethod
    def _get_solvers(
        system: CompositeSystem,
        held: tuple[object, str],
        scope: str = '',
    ) -> dict[str, tuple[object, str]]:
        """Return the solver of every system of a tree, with the path of its holder.

        The result maps the path of every system to its solver and the path of the system that gives it. A nested system
        keeps its own solver, as it keeps its own time, and one without a solver takes `held`, the solver of the system
        that holds it. The root takes its own solver when the compilation is given none.
        """
        own = getattr(system, 'solver', None)
        if own is not None and (scope or held[0] is None):
            held = (own, scope)
        solvers = {scope: held}
        for child in getattr(system, 'systems', ()):
            solvers.update(CensusPass._get_solvers(child, held, f'{scope}{child.label}/'))
        return solvers

    @staticmethod
    def _get_flowing(records: Sequence[_CensusRecordSpec]) -> frozenset[str]:
        """Return the paths of the elements and groups that flow.

        Rate or noise laws change their states, or equations read derivatives of them, as the equations of motion of
        a cart and its pendulum do.
        """
        rates = {record.owner for record in records if KernelsPass._get_kind_of(record.law) in (RateLaw, NoiseLaw)}
        return frozenset(rates.union(*(record.differentiated for record in records)))

    @staticmethod
    def _get_corrected(law: BaseLaw, owner: str, flowing: frozenset[str]) -> BaseLaw:
        """Return an update law of an element that flows as a correction after each step.

        A value corrected after every step of the solver, such as a wall reflecting `x` to `abs(x)` with
        `UpdateLaw(x, abs(x))` or a controller setting the input of a plant, is an update law of an element whose states
        rate or noise laws change. An update law of an element that only steps, such as a counter, or one that adds to
        other updates, reads neighbours, or runs in an order or a phase, stays an update law.

        Args:
            law:
                The law.

            owner:
                The path of the element or group whose state the law changes.

            flowing:
                The paths of the elements and groups whose states rate or noise laws change.

        Returns:
            law:
                The law, or the correction it states.
        """
        if not isinstance(law, UpdateLaw) or not isinstance(law.target, Expression | FreeModuleElement):
            return law
        plain = not law.increment and law.source is None and law.order is None and law.phase is None
        if not plain or law.neighborhood is not None or owner not in flowing:
            return law
        return BoundaryLaw(law.target, law.value, label=law.label).prepare(owner)

    @staticmethod
    def _get_renewal_group(spec: KernelSpec, law: RenewalLaw) -> str:
        """Return the path of the element or group whose states a renewal law resets.

        Without one, it is the component of the law.
        """
        targets = [item for target in (law.reset or {}) for item in components_of(target)]
        reference = spec.references.get(targets[0]) if targets else None
        return spec.owner if reference is None else reference.entry.name.rpartition('.')[0]

    @staticmethod
    def _get_special_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str | None:
        """Return the owner of a renewal, a rule, a matching, or a topology request.

        It is `None` for another law.
        """
        if isinstance(law, JumpLaw) and law.waiting is not None:
            return CensusPass._get_renewal_group(spec, RenewalLaw(law.waiting, law.reset))
        if isinstance(law, RenewalLaw):
            return CensusPass._get_renewal_group(spec, law)
        if isinstance(law, RuleLaw):
            paths = [path for value in spec.roles.values() for path in ([value] if isinstance(value, str) else value)]
            return next((path for path in paths if path in agents), next(iter(paths), spec.owner))
        if TopologyKernel._request_of(law) is not None:
            return spec.group or next(iter(spec.groups), spec.owner)
        return None

    @staticmethod
    def _is_offset(target: object) -> bool:
        """Return whether a target is a member at an offset, such as `shift_member(n, 1)`."""
        operator = target.operator() if isinstance(target, Expression) else None
        return getattr(operator, 'name', lambda: '')() == SHIFT

    @staticmethod
    def _get_changed_target(spec: KernelSpec, law: BaseLaw) -> object:
        """Return the target of a law, or the first state a jump changes.

        A jump changes the state of a member it reads at an offset, such as `shift_member(n, 1)`, of the group of its
        other states, so the first state read at no offset comes first.
        """
        if isinstance(law, JumpLaw):
            changed = list(law.changes or {})
            direct = [item for item in changed if not CensusPass._is_offset(item)]
            return next(iter(direct or changed))
        if not law._get_targets():
            error_msg = f'A {type(law).__name__} of {spec.owner} changes no state of its component.'
            raise ValidationError(error_msg)
        return law._get_targets()[0]

    @staticmethod
    def _get_law_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str:
        """Return the path of the element or group a law belongs to.

        A renewal belongs to the group whose states it resets, a rule or a matching to the first agent group among its
        roles, a topology request to the group it changes, and a lattice law to the element or group whose sites it
        updates. Flows, Monte Carlo samples, equations, and events belong to their component, and any other law to the
        element or group whose state it changes.

        Args:
            spec:
                The kernel spec of the law's component.

            law:
                The law.

            agents:
                The paths of the agent elements and groups.

        Returns:
            owner:
                The path of the owner.

        Raises:
            ValidationError:
                The law changes no state of its component, or targets a symbol that is not a state.
        """
        owner = CensusPass._get_special_owner(spec, law, agents)
        if owner is not None:
            return owner
        if isinstance(law, TransferLaw | ConstraintLaw | EventLaw) or (
            isinstance(law, RelaxationLaw) and not law._get_targets()
        ):
            return spec.owner
        lattice = isinstance(law, RelaxationLaw | LocalRuleLaw | SampleLaw)
        target = law._get_targets()[0] if lattice else CensusPass._get_changed_target(spec, law)
        return spec._get_target_references(target)[0].entry.name.rpartition('.')[0]

    @staticmethod
    def _get_expressions_of(law: BaseLaw) -> list[Expression]:
        """Return the symbolic components of the targets and values of a law.

        The laws it holds are included.
        """
        found: list[Expression] = []
        for value in (*law._get_targets(), *law._get_values()):
            if isinstance(value, BaseLaw):
                found += CensusPass._get_expressions_of(value)
            elif isinstance(value, Expression | FreeModuleElement):
                found += [SR(item) for item in components_of(value)]
        return found

    @staticmethod
    def _get_outside_reductions(expression: Expression) -> Expression:
        """Return an expression with every reduction replaced by one placeholder.

        Neighbourhood reductions are replaced too.
        """
        calls = [item.call for item in reductions_in(expression)] + [item.call for item in neighborhoods_in(expression)]
        calls += [item.call for item in hyperedges_in(expression)]
        if not calls:
            return expression
        placeholder = symbol('_reduction')
        return SR(expression).subs(dict.fromkeys(calls, placeholder))

    @staticmethod
    def _get_end_of(edges: KernelSpec, law: BaseLaw) -> str | None:
        """Return the end of an edge whose member alone a law reads.

        It is `None` for a law of the edge. A law of one end reads no member of the edge group and, outside its
        reductions, the symbols of one end only, and targets states of that end. Inside its reductions, the other end
        stands for the other members.

        Args:
            edges:
                The spec of the edges.

            law:
                The law.

        Returns:
            end:
                `'source'`, `'target'`, or `None`.
        """
        request = TopologyKernel._request_of(law)
        if request is not None and request.kind in EDGE_REQUESTS:
            return None
        if request is None and not all(isinstance(item, Expression | FreeModuleElement) for item in law._get_targets()):
            return None
        expressions = CensusPass._get_expressions_of(law)
        references = edges.references
        reads = [references.get(item) for item in symbols_of(expressions)]
        if any(reference is not None and reference.source is ReadKind.MEMBER for reference in reads):
            return None
        sides = {ReadKind.FIRST: 'source', ReadKind.SECOND: 'target'}
        outside = symbols_of(CensusPass._get_outside_reductions(item) for item in expressions)
        ends = {
            sides[reference.source]
            for item in outside
            if (reference := references.get(item)) is not None and reference.source in sides
        }
        if not ends and request is not None:
            return 'source'
        return next(iter(ends)) if len(ends) == 1 else None

    @staticmethod
    def _get_substituted_items(
        value: Mapping[object, object] | list[object] | tuple[object, ...] | CallSpec,
        mapping: Mapping[Expression, Expression],
    ) -> object:
        """Return a mapping, sequence, or call of a law with its item symbols replaced."""
        if isinstance(value, Mapping):
            return {
                CensusPass._get_substituted(key, mapping): CensusPass._get_substituted(item, mapping)
                for key, item in value.items()
            }
        if isinstance(value, CallSpec):
            if value.arguments is None:
                return value
            return replace(
                value,
                arguments=tuple(CensusPass._get_substituted(item, mapping) for item in value.arguments),
            )
        return type(value)(CensusPass._get_substituted(item, mapping) for item in value)

    @staticmethod
    def _get_substituted(value: object, mapping: Mapping[Expression, Expression]) -> object:
        """Return a value of a law with symbols replaced.

        The replacement goes through vectors, mappings, sequences, calls, and laws.
        """
        if isinstance(value, BaseLaw):
            parameters = value.get_params(deep=False)
            return BasePass._get_replaced(
                value,
                **{name: CensusPass._get_substituted(item, mapping) for name, item in parameters.items()},
            )
        if isinstance(value, FreeModuleElement):
            return vector(SR(item).subs(mapping) for item in value)
        if isinstance(value, Expression):
            return value.subs(mapping)
        if isinstance(value, Mapping | list | tuple | CallSpec):
            return CensusPass._get_substituted_items(value, mapping)
        return value

    @staticmethod
    def _get_end_law(edges: KernelSpec, law: BaseLaw, end: str) -> BaseLaw:
        """Return a law of one end of an edge written for the member.

        The other end stands for other members.

        Args:
            edges:
                The spec of the edges.

            law:
                The law, which reads one end outside its reductions.

            end:
                `'source'` or `'target'`.

        Returns:
            law:
                The checked law in the symbols of the member and of the other member.
        """
        own = ReadKind.FIRST if end == 'source' else ReadKind.SECOND
        mapping = {}
        for item, reference in edges.references.items():
            if reference.source in {ReadKind.FIRST, ReadKind.SECOND}:
                member, other = SpecsPass._get_end_symbols(item)
                mapping[item] = member if reference.source is own else other
        substituted = CensusPass._get_substituted(law, mapping)
        if not isinstance(substituted, BaseLaw):
            error_msg = f'The law {law!r} of {edges.owner} is not a law once written for the member.'
            raise TypeError(error_msg)
        return substituted.prepare(edges.owner)

    @staticmethod
    def _get_own_end(edges: KernelSpec, law: BaseLaw) -> str | None:
        """Return the end of an edge whose state a rate law per edge targets.

        It is `None` for another law.
        """
        if not isinstance(law, RateLaw) or not isinstance(law.value, Expression):
            return None
        target = law._get_targets()[0]
        reference = edges.references.get(target) if isinstance(target, Expression) else None
        if reference is None:
            return None
        return {ReadKind.FIRST: 'source', ReadKind.SECOND: 'target'}.get(reference.source)

    @staticmethod
    def _get_ends_of(edges: KernelSpec) -> dict[Expression, tuple[Expression, bool]]:
        """Return the symbol of the member and whether it is a state.

        They are given by the symbol of the other member.
        """
        found = {}
        for item, reference in edges.references.items():
            if reference.source in {ReadKind.FIRST, ReadKind.SECOND}:
                member, other = SpecsPass._get_end_symbols(item)
                found[other] = (member, reference.is_state)
        return found

    @staticmethod
    def _get_difference(edges: KernelSpec, value: Expression) -> tuple[Expression, Expression] | None:
        """Return the coefficient and state of a rate that scales the ends' difference.

        The rate is a constant times the difference. The coefficient reads no state and nothing of the other member.

        Args:
            edges:
                The spec of the edges.

            value:
                The rate of the member, in the symbols of the member and of the other member.

        Returns:
            difference:
                The coefficient and the symbol of the state of the member, or `None`.
        """
        ends = CensusPass._get_ends_of(edges)
        states = {member for member, is_state in ends.values() if is_state}
        found = [item for item in value.variables() if item in ends]
        if len(found) != 1 or not ends[found[0]][1]:
            return None
        other, member = found[0], ends[found[0]][0]
        coefficient = SR(value.diff(other))
        if any(item in ends or item in states for item in coefficient.variables()):
            return None
        if not SR(value - coefficient * (other - member)).expand().is_zero():
            return None
        return coefficient, member

    @staticmethod
    def _get_squared_spacing(place: FieldElement) -> float | None:
        """Return the factor that turns a field Laplacian into a sum over neighbours.

        The sum is of the differences over its neighbours. It is the squared spacing of a lattice with one spacing,
        `None` for a lattice with several, and 1 elsewhere.
        """
        space = place.space
        if not isinstance(space, LatticeSpace):
            return 1.0
        spacings = set(space.prepare().spacings_)
        return spacings.pop() ** 2 if len(spacings) == 1 else None

    @staticmethod
    def _get_shifted_law(edges: KernelSpec, law: RateLaw, end: str, space: LatticeSpace) -> RateLaw:
        """Return a rate per edge of a lattice field as a sum of shifted reads.

        The sum runs over the axes, and the reads are of the other end. The target of an edge is one site ahead of its
        source along an axis, so the other end of a rate of the target is one site behind, and that of a rate of the
        source one site ahead.

        Args:
            edges:
                The spec of the edges.

            law:
                The rate in the symbols of the member and of the other member.

            end:
                `'source'` or `'target'`, the end the rate belongs to.

            space:
                The lattice.

        Returns:
            law:
                The rate of the member.
        """
        value = SR(law.value)
        ends = {
            other: member for other, (member, _) in CensusPass._get_ends_of(edges).items() if other in value.variables()
        }
        sign = 1 if end == 'source' else -1
        dimensions = space.prepare().n_dim_
        total = SR(0)
        for axis, step in enumerate(space.spacings_):
            offset = [sign * step if index == axis else 0 for index in range(dimensions)]
            total += value.subs({other: shifted(member, offset) for other, member in ends.items()})
        return BasePass._get_replaced(law, value=total).prepare(edges.owner)

    @staticmethod
    def _get_field_laws(edges: KernelSpec, laws: list[BaseLaw]) -> tuple[list[BaseLaw], list[BaseLaw]]:
        """Return the rates per edge of a field lowered to member laws, and the rest.

        The rest are the laws that stay per edge. Two rates of both ends that are the same constant times the difference
        of the ends lower to the field's Laplacian times its squared spacing. Other rates on a lattice lower to
        whole-site shifted reads.

        Args:
            edges:
                The spec of the edges, whose `place` is the field.

            laws:
                The laws per edge.

        Returns:
            laws:
                The laws of the member, and the laws that stay per edge.
        """
        place = edges.place
        if place is None:
            return [], laws
        written: dict[int, tuple[str, RateLaw]] = {}
        for index, law in enumerate(laws):
            end = CensusPass._get_own_end(edges, law)
            item = CensusPass._get_end_law(edges, law, end) if end is not None else None
            if end is not None and isinstance(item, RateLaw):
                written[index] = (end, item)
        scale = CensusPass._get_squared_spacing(place)
        lowered: list[BaseLaw] = []
        taken: set[int] = set()
        for index, (end, law) in written.items():
            partner = next(
                (
                    other
                    for other, (side, item) in written.items()
                    if other not in taken
                    and side != end
                    and str(item._get_targets()[0]) == str(law._get_targets()[0])
                    and SR(SR(item.value) - SR(law.value)).is_zero()
                ),
                None,
            )
            difference = CensusPass._get_difference(edges, SR(law.value))
            if index in taken or partner is None or difference is None or scale is None:
                continue
            coefficient, member = difference
            lowered.append(
                BasePass._get_replaced(law, value=coefficient * scale * laplacian(member)).prepare(edges.owner),
            )
            taken |= {index, partner}
        if isinstance(place.space, LatticeSpace):
            lowered += [
                CensusPass._get_shifted_law(edges, law, end, place.space)
                for index, (end, law) in written.items()
                if index not in taken
            ]
            taken |= set(written)
        return lowered, [law for index, law in enumerate(laws) if index not in taken]

    @staticmethod
    def _get_targets_second(spec: KernelSpec, law: BaseLaw) -> bool:
        """Check whether a hyperedge interaction law targets the member of its group.

        The group is on hyperedges or edges.
        """
        path = spec.roles.get('hyperedge', spec.roles.get('edge'))
        targets = [item for item in law._get_targets() if isinstance(item, Expression | FreeModuleElement)]
        return any(
            reference.entry.name.rpartition('.')[0] == path
            for target in targets
            for reference in spec._get_target_references(target)
        )

    @staticmethod
    def _get_hyperedge_laws(spec: KernelSpec, laws: list[BaseLaw]) -> list[BaseLaw]:
        """Return the laws of one kind of a hyperedge interaction.

        Their reductions over the members are written out. The spec of the focal member keeps the laws of the nodes, and
        the spec of the hyperedge the laws that target the member of the group on hyperedges or edges.

        Args:
            spec:
                The spec of the hyperedge interaction for one kind of laws and one size.

            laws:
                The declared laws.

        Returns:
            laws:
                The laws of the spec, checked.

        Raises:
            TypeError:
                A law is no longer a law once its reductions are written out.
        """
        kept = []
        for law in laws:
            if CensusPass._get_targets_second(spec, law) != (spec.hyperedge == HYPEREDGE_LAWS):
                continue
            expansions = HyperedgeTables._get_expansions(spec, CensusPass._get_expressions_of(law))
            expanded = CensusPass._get_substituted(law, expansions) if expansions else law
            if not isinstance(expanded, BaseLaw):
                error_msg = f'The law {law!r} of {spec.owner} is not a law once its reductions are written out.'
                raise TypeError(error_msg)
            kept.append(HyperedgeTables._get_masked(spec, expanded.prepare(spec.owner)))
        return kept

    @staticmethod
    def _get_opaque_calls(expression: Expression) -> dict[Expression, Expression]:
        """Return a symbol per outermost call of an unknown function or library operator.

        Such a call has no value at numbers.
        """
        if isinstance(expression.operator(), SymbolicFunction | _OperatorFunction):
            return {expression: SR.symbol(f'_call_{abs(hash(str(expression)))}')}
        found: dict[Expression, Expression] = {}
        for operand in expression.operands():
            found.update(CensusPass._get_opaque_calls(operand))
        return found

    @staticmethod
    def _breaks_symmetry(residual: Expression) -> bool:
        """Check whether an expression a symmetry needs to be 0 is not 0.

        It is evaluated at random values of its symbols.
        """
        opaque = residual.subs(CensusPass._get_opaque_calls(residual))
        symbols = opaque.variables()
        generator = np.random.default_rng(len(symbols))
        for _ in range(SYMMETRY_TRIALS):
            values = dict(zip(symbols, generator.uniform(0.1, 1.0, len(symbols)).tolist(), strict=True))
            value = opaque.subs(values)
            if not value.is_numeric() or abs(complex(value)) > SYMMETRY_TOLERANCE:
                return bool(value.is_numeric())
        return False

    @staticmethod
    def _check_tensor_rate(law: RateLaw, name: str, tensor: TensorSpec, value: Matrix) -> None:
        """Raise an error for a matrix rate with inconsistent dependent entries.

        Its dependent entries are not the ones its independent entries give.
        """
        expected = tensor.get_matrix_of(value[row, column] for row, column in tensor.indices)
        size = tensor.n_dim
        for row in range(size):
            for column in range(size):
                residual = distribute_operators(SR(value[row, column] - expected[row, column])).expand()
                if residual.is_zero() or residual.simplify_full().is_zero():
                    continue
                if not CensusPass._breaks_symmetry(residual):
                    continue
                last = row == column == size - 1 and tensor.symmetry == 'traceless'
                where = f'the trace {value.trace()}' if last else f'the entry [{row}, {column}]'
                error_msg = (
                    f'The rate law {law.label or repr(law.value)} on the tensor state {name} breaks its symmetry '
                    f'{tensor.symmetry!r} at {where}, by the comparison at random component values, since Sage does '
                    'not simplify the difference to 0.'
                )
                raise TensorSymmetryError(error_msg)

    @staticmethod
    def _get_array_update(law: BaseLaw) -> BaseLaw:
        """Return the array update of an update law of a whole array, or the law itself.

        Such a law, such as `UpdateLaw(s.target.height, drop)`, gives a NumPy function of the whole array its state.
        """
        function = law.value if isinstance(law, UpdateLaw) else None
        if not isinstance(law, UpdateLaw) or not law.arrays or not callable(function):
            return law
        return ArrayUpdateLaw(law.target, function, label=law.label, score=law.score)

    @staticmethod
    def _get_relaxation(law: BaseLaw) -> BaseLaw:
        """Return the relaxation law of an update law of a relaxation, or the law itself.

        A rule of the relaxation that is an update law with a neighbourhood runs as the local rule it stands for, and
        a drive or a rule that updates a whole array by a function as an array update.
        """
        relaxation = relaxation_of(law.value) if isinstance(law, UpdateLaw) else None
        if relaxation is None:
            return CensusPass._get_array_update(law)
        rules = tuple(
            (
                rule._get_local_rule()
                if isinstance(rule, UpdateLaw) and rule.neighborhood is not None
                else CensusPass._get_array_update(rule)
            )
            for rule in relaxation.rules
        )
        return RelaxationLaw(
            tuple(CensusPass._get_array_update(item) for item in relaxation.drive),
            rules,
            relaxation.stable,
            relaxation.max_iterations,
            relaxation.activity,
            label=law.label,
        )

    @staticmethod
    def _get_component_rates(spec: KernelSpec, law: BaseLaw) -> list[BaseLaw]:
        """Return a rate law on a tensor or vector state as one rate per component.

        The vector rate is a field operator rate.

        Args:
            spec:
                The kernel spec of the component.

            law:
                The law as the component returns it.

        Returns:
            laws:
                The rate of every independent component, or any other law as it is.

        Raises:
            TensorShapeError:
                The target is not the matrix of a tensor state, or the rate does not have its shape.

            TensorSymmetryError:
                The rate breaks the symmetry of the state.
        """
        if not isinstance(law, RateLaw):
            return [law]
        if isinstance(law.target, FreeModuleElement) and isinstance(law.value, FreeModuleElement):
            if not any(FieldLowering._has_operators(item) for item in law.value):
                return [law]
            pairs = zip(law.target, law.value, strict=True)
            return [RateLaw(target, rate, law.order, label=law.label) for target, rate in pairs]
        if not isinstance(law.target, Matrix):
            return [law]
        references = [spec.references.get(item) for item in law.target.variables()]
        entry = next((item.entry for item in references if item is not None and item.entry.tensor is not None), None)
        if entry is None or entry.tensor is None:
            error_msg = f'A rate law of {spec.owner} targets the matrix {law.target}, which is no tensor state.'
            raise TensorShapeError(error_msg)
        tensor = entry.tensor
        targets = [law.target[row, column] for row, column in tensor.indices]
        if BaseLaw._is_function(law.value):
            return [RateLaw(vector(targets), law.value, law.order, label=law.label)]
        if not isinstance(law.value, Matrix) or law.value.dimensions() != law.target.dimensions():
            error_msg = (
                f'A rate law of {spec.owner} on the tensor state {entry.name} has the value {law.value}. Give a matrix '
                f'of {tensor.n_dim} rows and {tensor.n_dim} columns.'
            )
            raise TensorShapeError(error_msg)
        CensusPass._check_tensor_rate(law, entry.name, tensor, law.value)
        rates = [law.value[row, column] for row, column in tensor.indices]
        return [RateLaw(target, rate, law.order, label=law.label) for target, rate in zip(targets, rates, strict=True)]

    @staticmethod
    def _get_role_of(namespace: SimpleNamespace, holder: SimpleNamespace) -> str | None:
        """Return the name in a namespace that holds a namespace or a copy of it.

        It holds it directly or inside, and the result is `None` otherwise.
        """
        for name, value in vars(namespace).items():
            for item in value if isinstance(value, tuple) else (value,):
                if not isinstance(item, SimpleNamespace):
                    continue
                if item is holder or vars(item).keys() == vars(holder).keys() or CensusPass._get_role_of(item, holder):
                    return name
        return None

    @staticmethod
    def _get_returned_laws(spec: KernelSpec, component: BaseComponent) -> Sequence[BaseComponent]:
        """Return the laws a component returns.

        A read of a state a role does not have raises an error of the role.

        Args:
            spec:
                The kernel spec of the component.

            component:
                The object whose `laws` method describes the component.

        Returns:
            laws:
                What the component returns.

        Raises:
            RoleElementError:
                An interaction reads a name that the element bound to a role does not have.
        """
        try:
            return component._get_prepared_laws(spec.namespace)
        except AttributeError as error:
            holder = error.obj
            if not isinstance(component, BaseInteraction) or not isinstance(holder, SimpleNamespace):
                raise
            role = CensusPass._get_role_of(spec.namespace, holder)
            if role is None:
                raise
            names = [name for name, value in vars(holder).items() if not isinstance(value, SimpleNamespace | tuple)]
            carried = ', '.join(sorted(names)) or 'nothing'
            error_msg = (
                f'The role {role!r} of {spec.owner} reads the state {error.name!r}, but the element bound to it '
                f'carries {carried}.'
            )
            raise RoleElementError(error_msg) from error

    @staticmethod
    def _get_declared_laws(
        spec: KernelSpec,
        component: BaseComponent,
        solver: object = None,
        scope: str = '',
    ) -> list[BaseLaw]:
        """Return the laws a component returns for its namespace.

        It first checks their classes against its binding.

        Args:
            spec:
                The kernel spec of the component.

            component:
                The object whose `laws` method describes the component.

            solver:
                The solver of the compilation, whose moves turn objectives of members into sweeps, or `None`.

            scope:
                The path of the system that holds the solver, which the names of its moves are relative to.

        Returns:
            laws:
                The laws as declared.

        Raises:
            ValidationError:
                `laws` returns something other than declarable laws, or laws its binding does not take.
        """
        returned = CensusPass._get_returned_laws(spec, component)
        laws = [law for law in returned if isinstance(law, BaseLaw)]
        if not isinstance(returned, list | tuple) or len(laws) < len(returned):
            error_msg = f'`_get_laws` of {spec.owner} must return a list of laws.'
            raise ValidationError(error_msg)
        laws = [CensusPass._get_sampled(spec, law, solver, scope) for law in laws]
        laws = [item for law in laws for item in CensusPass._get_mean_jumps(law, solver)]
        lowered = [BilinearLaw._get_lowered(law) for law in CensusPass._get_programs(laws, spec.owner)]
        split = [item for law in lowered for item in CensusPass._get_component_rates(spec, law)]
        declared = [CensusPass._get_relaxation(law).prepare(spec.owner) for law in split]
        if spec.hyperedge is not None:
            declared = CensusPass._get_hyperedge_laws(spec, declared)
        searched = spec.pairs is not None and spec.edges is None
        alone = spec.edges is not None and spec.edges.pairs is None
        rules = [law for law in declared if (searched or alone) and isinstance(law, RuleLaw)]
        declared = [law for law in declared if law not in rules]
        if isinstance(spec.pairs, PairSearch) and not all(isinstance(law, RateLaw) for law in declared):
            error_msg = (
                f'Pair interaction {spec.owner} returns laws other than Rate, which pair interactions do not take.'
            )
            raise ValidationError(error_msg)
        if isinstance(spec.pairs, EdgeSearch):
            declared = CensusPass._get_field_laws(
                spec,
                [law for law in declared if CensusPass._get_end_of(spec, law) is None],
            )[1]
        if spec.edges is not None:
            edges = spec.edges
            lowered, kept = CensusPass._get_field_laws(
                edges,
                [law for law in declared if CensusPass._get_end_of(edges, law) is None],
            )
            if edges.pairs is None and kept:
                error_msg = (
                    f'Interaction {spec.owner} over neighbours has laws per edge, but its members live on a space '
                    'without edges. Give the group a lattice or network space, write laws of one end, or on a field '
                    'write a constant times the difference of the ends.'
                )
                raise ValidationError(error_msg)
            declared = [
                *(
                    CensusPass._get_end_law(edges, law, end)
                    for law in declared
                    if (end := CensusPass._get_end_of(edges, law)) is not None
                ),
                *lowered,
            ]
        edge_kinds = (RateLaw, NoiseLaw, JumpLaw, UpdateLaw, BoundaryLaw)
        if isinstance(spec.pairs, EdgeSearch) and not all(isinstance(law, edge_kinds) for law in declared):
            error_msg = (
                f'Interaction {spec.owner} over neighbours returns laws other than Rate, Noise, Jump, Update, or '
                'Boundary laws for its edges. Its edges take these laws, and the laws that read one end of an edge '
                'run once per member.'
            )
            raise ValidationError(error_msg)
        checked = []
        for law in declared:
            try:
                checked.append(law.prepare(spec.owner))
            except ValidationError as error:
                error_msg = f'{type(law).__name__} of {spec.owner}: {error}'
                raise ValidationError(error_msg) from error
        return [*rules, *checked]

    @staticmethod
    def _get_programs(laws: Sequence[BaseLaw], owner: str) -> list[BaseLaw]:
        """Return the laws of a component, its linear objectives as linear programs.

        An objective without controls that is linear in its targets, or a sum of their absolute values, is minimised
        as a linear program, under the constraints of the component that read its targets, such as the balances and the
        bounds of the fluxes of a metabolic network. A sequence of values is minimised in order, each minimum held
        while the next one is sought, as the smallest total flux at the greatest growth. An inequality of another
        component is a constraint of the program of the value its `target` names, such as the uptake a medium allows.

        Raises:
            ValidationError:
                A sequence of values is not linear in the targets, or an inequality outside a program names no target.
        """
        programs = {
            id(law): law
            for law in laws
            if isinstance(law, ObjectiveLaw) and law.controls is None and CensusPass._is_linear_objective(law)
        }
        targets = {str(item) for law in programs.values() for item in law._get_targets()}
        kept: list[BaseLaw] = []
        constraints: list[Expression] = []
        for law in laws:
            if id(law) in programs:
                continue
            objective = isinstance(law, ObjectiveLaw) and law.controls is None
            if objective and not isinstance(law.value, Expression) and isinstance(law.value, Sequence):
                error_msg = (
                    f'The objective of {owner} is a sequence of values, which is minimised in order as a linear '
                    'program, so every value is linear in the targets or a sum of their absolute values.'
                )
                raise ValidationError(error_msg)
            if not isinstance(law, ConstraintLaw):
                kept.append(law)
                continue
            relation = SR(law.lhs) if law.rhs is None else SR(law.lhs) == SR(law.rhs)
            inequality = relation.operator() is not operator.eq
            if targets & {str(item) for item in relation.variables()}:
                constraints.append(relation)
            elif inequality and law.target is None:
                error_msg = (
                    f'The inequality {relation} of {owner} holds in a linear program, so it names the value it bounds '
                    'in `target`, such as `ConstraintLaw(flux >= -uptake, target=flux)`.'
                )
                raise ValidationError(error_msg)
            elif inequality:
                kept.append(LinearProgramLaw([SR(law.target)], [relation], label=law.label))
            else:
                kept.append(law)
        converted = [
            LinearProgramLaw(
                list(law._get_targets()),
                constraints if position == 0 else [],
                [LinearObjectiveSpec(value, sense='min') for value in law._get_values_in_order()],
                active=law.active,
                infeasible=law.infeasible,
                duals=law.duals,
                label=law.label,
            )
            for position, law in enumerate(programs.values())
        ]
        return [*kept, *converted]

    @staticmethod
    def _is_linear_objective(law: ObjectiveLaw) -> bool:
        """Return whether an objective is linear or sums absolute values of its targets."""
        targets = {str(item) for item in law._get_targets()}
        for value in law._get_values_in_order():
            expression = SR(value)
            absolutes = {
                item: symbol(f'_absolute_{index}')
                for index, item in enumerate(BasePass._get_calls_of(expression, 'abs'))
                if str(item.operands()[0]) in targets
            }
            expression = expression.subs(absolutes)
            for item in expression.variables():
                if str(item) in targets and targets & {str(other) for other in expression.diff(item).variables()}:
                    return False
        return True

    @staticmethod
    def _get_sampled(spec: KernelSpec, law: BaseLaw, solver: object, scope: str = '') -> BaseLaw:
        """Return the objective of members a Metropolis solver moves, as a sweep.

        The objective is the local objective of a member, such as its energy over the temperature, and the moves, the
        rule, the schedule, and the neighbourhood are the settings of the solver, so the sweep accepts a move with the
        weight `exp(-change)` of the objective. The moves name states relative to `scope`, the path of the system that
        holds the solver.
        """
        metropolis = getattr(solver, 'metropolis', solver)
        moves = getattr(metropolis, 'moves', None)
        if not isinstance(law, ObjectiveLaw) or not moves or not isinstance(law.targets, Expression):
            return law
        reference = spec.references.get(law.targets)
        name = reference.entry.name.removeprefix(scope) if reference is not None else None
        if name not in moves:
            return law
        kind = moves[name]
        if callable(kind):
            move = MoveSpec(law.targets, 'function', function=kind)
        else:
            values = tuple((getattr(metropolis, 'values', None) or {}).get(name, ()))
            move = MoveSpec(law.targets, kind, values, getattr(metropolis, 'width', 1.0))
        over = getattr(metropolis, 'neighborhood', None) or RadiusNeighborhood(1, metric='manhattan')
        sampler = SamplerSpec(
            getattr(metropolis, 'rule', 'metropolis'),
            1.0,
            getattr(metropolis, 'schedule', 'checkerboard'),
        )
        label = law.label or spec.owner.rpartition('/')[2]
        return SampleLaw(ObjectiveSpec(law.value, over), move, sampler, label=label)

    @staticmethod
    def _get_mean_jumps(law: BaseLaw, solver: object) -> list[BaseLaw]:
        """Return a jump as the rate laws of its mean under an ODE solver, else as it is.

        A jump that adds `changes` at the rate `r`, such as `JumpLaw(k * a, {a: -1, b: 1})`, moves every state by its
        change times `r` on average, so an ODE solver runs `RateLaw(a, -k * a)` and `RateLaw(b, k * a)`. A labelled jump
        whose changes are `-1` and `+1` is a flow of that label.
        """
        get_form = getattr(type(solver), '_get_form', None)
        ode = callable(get_form) and get_form() == FormKind.ODE
        if not ode or not isinstance(law, JumpLaw) or law.waiting is not None or not isinstance(law.rate, Expression):
            return [law]
        changes = law.changes or {}
        if law.label and all(change in (-1, 1) for change in changes.values()):
            return list(FlowTermLaw._get_terms(RateLaw(dict(changes), law.rate, label=law.label), law.label))
        return [RateLaw(target, SR(change) * law.rate) for target, change in changes.items()]

    @staticmethod
    def _get_component_of(node: _NodeSpec) -> BaseComponent:
        """Return the component whose laws describe a node, a member for a group."""
        if isinstance(node.component, GroupElement):
            group = node.component
            return group.element(group.label).set_params(**(group.values or {})).prepare()
        if not isinstance(node.component, BaseComponent):
            error_msg = f'The node {node.path} holds a {type(node.component).__name__}, which declares no laws.'
            raise TypeError(error_msg)
        return node.component
