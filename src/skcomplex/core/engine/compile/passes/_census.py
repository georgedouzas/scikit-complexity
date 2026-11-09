"""Take the census of the laws every component of a system declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix
from sage.symbolic.function import SymbolicFunction

from ....components.base import BasePart
from ....components.elements import FieldElement, GroupElement
from ....components.interactions import BaseInteraction
from ....components.laws import (
    BaseLaw,
    BoundaryLaw,
    CascadeLaw,
    CrossingLaw,
    EquationLaw,
    ExtremalLaw,
    JumpLaw,
    LocalRuleLaw,
    MatchLaw,
    NoiseLaw,
    RateLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
    UpdateLaw,
)
from ....components.laws.utils import laplacian, shifted
from ....components.laws.utils._reductions import hyperedges_in, neighborhoods_in, reductions_in
from ....components.laws.utils._tensors import distribute_operators
from ....components.spaces import CellComplexSpace, LatticeSpace
from ....components.specs import CensusRecordSpec, CensusSpec, NodeSpec
from ....components.systems import CompositeSystem
from ....errors import RoleElementError, SkcomplexError, TensorShapeError, TensorSymmetryError, ValidationError
from ....specs import CallSpec, EntrySpec, TensorSpec
from ....types import ElementKind, ProcessKey, VariableKind
from ..._utils import components_of, declarations_of, symbol, symbols_of, vector
from ...problems._block_spec import BlockSpec
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._edge_search import EdgeSearch
from ...problems._kernel_spec import KernelSpec
from ...problems._pair_search import PairSearch
from ...problems._topology_kernel import TopologyKernel
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation
from .._field_lowering import FieldLowering
from .._observables import ObservableEvaluator
from .._observer import Observer
from .._renewal_law import RenewalLaw
from ._cells import DEGREES
from ._hyperedges import HYPEREDGE_LAWS, HyperedgeTables
from ._layout import LayoutPass
from ._noise import NoisePass
from ._specs import SpecsPass

EDGE_REQUESTS = ('remove-edge', 'rewire')
SYMMETRY_TOLERANCE = 1e-9
SYMMETRY_TRIALS = 3


class CensusPass(BasePass):
    """Pass that takes the census of the laws of the system.

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
        compilation.census = self._census_of(compilation.system, Compilation._required(compilation.nodes, 'nodes'))

    @staticmethod
    def _state_reads(spec: KernelSpec, value: object) -> frozenset[tuple[str, int]]:
        """Return the state components that a law value depends on."""
        if not isinstance(value, Expression):
            return frozenset()
        return frozenset(
            (reference.entry.name, reference.column)
            for item in value.variables()
            if (reference := spec.references.get(item)) is not None and reference.is_state
        )

    @staticmethod
    def _noise_census(
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
            tuple(NoisePass._process_keys(spec, law, declarations)),
            tuple((item.entry.name, item.column) for item in references if item is not None),
            tuple(CensusPass._state_reads(spec, value) for value in values),
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
    def _component_kind(node: NodeSpec) -> ElementKind:
        """Return the kind of the component of a node."""
        if isinstance(node.component, GroupElement) and node.component.variable:
            return ElementKind.VARIABLE_GROUP
        if isinstance(node.component, GroupElement):
            parts = {'sites': ElementKind.SITE_GROUP, 'edges': ElementKind.EDGE_GROUP}
            parts['hyperedges'] = ElementKind.HYPEREDGE_GROUP
            return parts.get(node.component.on or '', ElementKind.GROUP)
        return ElementKind.ELEMENT

    @staticmethod
    def _equation_targets(spec: KernelSpec, law: BaseLaw) -> tuple[EntrySpec, ...]:
        """Return the entry of the state an equation names as its target.

        It is none for any other law or target.
        """
        target = law.target if isinstance(law, EquationLaw) else None
        reference = spec.references.get(target) if isinstance(target, Expression) else None
        return (reference.entry,) if reference is not None and reference.is_state else ()

    @staticmethod
    def _cell_degrees(spec: KernelSpec, declarations: DeclarationsSpec) -> dict[str, int]:
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
    def _records_of(
        block: BlockSpec,
        elements: dict[str, BlockSpec],
        observer: Observer,
        declarations: DeclarationsSpec,
        agents: set[str],
    ) -> list[CensusRecordSpec]:
        """Return the census records of the laws of one component."""
        records = []
        specs, _ = SpecsPass._specs_of(block, elements, declarations, observer)
        for spec in specs:
            component = CensusPass._component_of(block.node)
            for law in CensusPass._declared_laws(spec, component):
                items = [item for target in law._get_targets() for item in components_of(target)]
                references = [spec.references.get(item) for item in items if isinstance(item, Expression)]
                targets = tuple(dict.fromkeys(item.entry for item in references if item is not None))
                targets = targets or CensusPass._equation_targets(spec, law)
                owner = CensusPass._law_owner(spec, law, agents)
                vectors = frozenset(entry.name for entry in targets if CensusPass._is_vector(entry, elements))
                kind = CensusPass._component_kind(elements.get(owner, block).node)
                processes, driven, reads = CensusPass._noise_census(spec, law, declarations)
                records.append(
                    CensusRecordSpec(
                        block.node.path,
                        owner,
                        law,
                        targets,
                        vectors,
                        kind,
                        SpecsPass._group_space(elements.get(owner, block)),
                        members=spec.rows,
                        handles=CensusPass._handle_uses(law, declarations.handles),
                        degrees=CensusPass._cell_degrees(spec, declarations),
                        processes=processes,
                        driven=driven,
                        reads=reads,
                    ),
                )
        return records

    @staticmethod
    def _census_of(system: CompositeSystem, nodes: list[NodeSpec]) -> CensusSpec:
        """Return the census of the laws of a system.

        The errors of the components that fail are included. The census declares every component and collects its laws
        as compilation does, without lowering them, and keeps going past a failing component, so validation reports
        every problem at once.

        Args:
            system:
                The system, with the observables of the run.

            nodes:
                The nodes of the system.

        Returns:
            census:
                The census.
        """
        declarations = DeclarationsSpec(time=symbol('t'))
        errors: list[tuple[str, SkcomplexError]] = []
        blocks = []
        groups = LayoutPass._group_sizes(nodes)
        for node in nodes:
            try:
                declarations, block = LayoutPass._declare_component(declarations, node, groups)
            except SkcomplexError as error:
                errors.append((node.path, error))
                continue
            blocks.append(block)
        elements = {block.node.path: block for block in blocks if not block.node.interaction}
        elements.update(declarations.processes)
        live = ObservableEvaluator._lowerable_observables(system, nodes)
        observer = Observer(declarations, live).prepare()
        agents = {block.node.path for block in blocks if block.node.is_agent}
        records = []
        for block in blocks:
            try:
                records += CensusPass._records_of(block, elements, observer, declarations, agents)
            except SkcomplexError as error:
                errors.append((block.node.path, error))
        return CensusSpec(system, records, errors, declarations.agents)

    @staticmethod
    def _renewal_group(spec: KernelSpec, law: RenewalLaw) -> str:
        """Return the path of the element or group whose states a renewal law resets.

        Without one, it is the component of the law.
        """
        targets = [item for target in (law.reset or {}) for item in components_of(target)]
        reference = spec.references.get(targets[0]) if targets else None
        return spec.owner if reference is None else reference.entry.name.rpartition('.')[0]

    @staticmethod
    def _special_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str | None:
        """Return the owner of a renewal, a rule, a matching, or a topology request.

        It is `None` for another law.
        """
        if isinstance(law, JumpLaw) and law.waiting is not None:
            return CensusPass._renewal_group(spec, RenewalLaw(law.waiting, law.reset))
        if isinstance(law, RenewalLaw):
            return CensusPass._renewal_group(spec, law)
        if isinstance(law, RuleLaw | MatchLaw):
            paths = [path for value in spec.roles.values() for path in ([value] if isinstance(value, str) else value)]
            return next((path for path in paths if path in agents), next(iter(paths), spec.owner))
        if TopologyKernel._request_of(law) is not None:
            return spec.group or next(iter(spec.groups), spec.owner)
        return None

    @staticmethod
    def _changed_target(spec: KernelSpec, law: BaseLaw) -> object:
        """Return the target of a law, or the first state a jump changes."""
        if isinstance(law, JumpLaw):
            return next(iter(law.changes or {}))
        if not law._get_targets():
            error_msg = f'A {type(law).__name__} of {spec.owner} changes no state of its component.'
            raise ValidationError(error_msg)
        return law._get_targets()[0]

    @staticmethod
    def _law_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str:
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
        owner = CensusPass._special_owner(spec, law, agents)
        if owner is not None:
            return owner
        if isinstance(law, TransferLaw | EquationLaw | CrossingLaw) or (
            isinstance(law, CascadeLaw) and not law._get_targets()
        ):
            return spec.owner
        lattice = isinstance(law, CascadeLaw | LocalRuleLaw | ExtremalLaw | SampleLaw)
        target = law._get_targets()[0] if lattice else CensusPass._changed_target(spec, law)
        return spec._target_references(target)[0].entry.name.rpartition('.')[0]

    @staticmethod
    def _expressions_of(law: BaseLaw) -> list[Expression]:
        """Return the symbolic components of the targets and values of a law.

        The laws it holds are included.
        """
        found: list[Expression] = []
        for value in (*law._get_targets(), *law._get_values()):
            if isinstance(value, BaseLaw):
                found += CensusPass._expressions_of(value)
            elif isinstance(value, Expression | FreeModuleElement):
                found += [SR(item) for item in components_of(value)]
        return found

    @staticmethod
    def _outside_reductions(expression: Expression) -> Expression:
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
    def _end_of(edges: KernelSpec, law: BaseLaw) -> str | None:
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
        expressions = CensusPass._expressions_of(law)
        references = edges.references
        reads = [references.get(item) for item in symbols_of(expressions)]
        if any(reference is not None and reference.source is ReadKind.MEMBER for reference in reads):
            return None
        sides = {ReadKind.FIRST: 'source', ReadKind.SECOND: 'target'}
        outside = symbols_of(CensusPass._outside_reductions(item) for item in expressions)
        ends = {
            sides[reference.source]
            for item in outside
            if (reference := references.get(item)) is not None and reference.source in sides
        }
        if not ends and request is not None:
            return 'source'
        return next(iter(ends)) if len(ends) == 1 else None

    @staticmethod
    def _substituted_items(
        value: Mapping[object, object] | list[object] | tuple[object, ...] | CallSpec,
        mapping: Mapping[Expression, Expression],
    ) -> object:
        """Return a mapping, sequence, or call of a law with its item symbols replaced."""
        if isinstance(value, Mapping):
            return {
                CensusPass._substituted(key, mapping): CensusPass._substituted(item, mapping)
                for key, item in value.items()
            }
        if isinstance(value, CallSpec):
            if value.arguments is None:
                return value
            return replace(value, arguments=tuple(CensusPass._substituted(item, mapping) for item in value.arguments))
        return type(value)(CensusPass._substituted(item, mapping) for item in value)

    @staticmethod
    def _substituted(value: object, mapping: Mapping[Expression, Expression]) -> object:
        """Return a value of a law with symbols replaced.

        The replacement goes through vectors, mappings, sequences, calls, and laws.
        """
        if isinstance(value, BaseLaw):
            parameters = value.get_params(deep=False)
            return BasePass._replaced(
                value,
                **{name: CensusPass._substituted(item, mapping) for name, item in parameters.items()},
            )
        if isinstance(value, FreeModuleElement):
            return vector(SR(item).subs(mapping) for item in value)
        if isinstance(value, Expression):
            return value.subs(mapping)
        if isinstance(value, Mapping | list | tuple | CallSpec):
            return CensusPass._substituted_items(value, mapping)
        return value

    @staticmethod
    def _end_law(edges: KernelSpec, law: BaseLaw, end: str) -> BaseLaw:
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
                member, other = SpecsPass._end_symbols(item)
                mapping[item] = member if reference.source is own else other
        substituted = CensusPass._substituted(law, mapping)
        if not isinstance(substituted, BaseLaw):
            error_msg = f'The law {law!r} of {edges.owner} is not a law once written for the member.'
            raise TypeError(error_msg)
        return substituted.prepare(edges.owner)

    @staticmethod
    def _own_end(edges: KernelSpec, law: BaseLaw) -> str | None:
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
    def _ends_of(edges: KernelSpec) -> dict[Expression, tuple[Expression, bool]]:
        """Return the symbol of the member and whether it is a state.

        They are given by the symbol of the other member.
        """
        found = {}
        for item, reference in edges.references.items():
            if reference.source in {ReadKind.FIRST, ReadKind.SECOND}:
                member, other = SpecsPass._end_symbols(item)
                found[other] = (member, reference.is_state)
        return found

    @staticmethod
    def _difference(edges: KernelSpec, value: Expression) -> tuple[Expression, Expression] | None:
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
        ends = CensusPass._ends_of(edges)
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
    def _squared_spacing(place: FieldElement) -> float | None:
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
    def _shifted_law(edges: KernelSpec, law: RateLaw, end: str, space: LatticeSpace) -> RateLaw:
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
            other: member for other, (member, _) in CensusPass._ends_of(edges).items() if other in value.variables()
        }
        sign = 1 if end == 'source' else -1
        dimensions = space.prepare().n_dim_
        total = SR(0)
        for axis, step in enumerate(space.spacings_):
            offset = [sign * step if index == axis else 0 for index in range(dimensions)]
            total += value.subs({other: shifted(member, offset) for other, member in ends.items()})
        return BasePass._replaced(law, value=total).prepare(edges.owner)

    @staticmethod
    def _field_laws(edges: KernelSpec, laws: list[BaseLaw]) -> tuple[list[BaseLaw], list[BaseLaw]]:
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
            end = CensusPass._own_end(edges, law)
            item = CensusPass._end_law(edges, law, end) if end is not None else None
            if end is not None and isinstance(item, RateLaw):
                written[index] = (end, item)
        scale = CensusPass._squared_spacing(place)
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
            difference = CensusPass._difference(edges, SR(law.value))
            if index in taken or partner is None or difference is None or scale is None:
                continue
            coefficient, member = difference
            lowered.append(BasePass._replaced(law, value=coefficient * scale * laplacian(member)).prepare(edges.owner))
            taken |= {index, partner}
        if isinstance(place.space, LatticeSpace):
            lowered += [
                CensusPass._shifted_law(edges, law, end, place.space)
                for index, (end, law) in written.items()
                if index not in taken
            ]
            taken |= set(written)
        return lowered, [law for index, law in enumerate(laws) if index not in taken]

    @staticmethod
    def _targets_second(spec: KernelSpec, law: BaseLaw) -> bool:
        """Check whether a hyperedge interaction law targets the member of its group.

        The group is on hyperedges or edges.
        """
        path = spec.roles.get('hyperedge', spec.roles.get('edge'))
        targets = [item for item in law._get_targets() if isinstance(item, Expression | FreeModuleElement)]
        return any(
            reference.entry.name.rpartition('.')[0] == path
            for target in targets
            for reference in spec._target_references(target)
        )

    @staticmethod
    def _hyperedge_laws(spec: KernelSpec, laws: list[BaseLaw]) -> list[BaseLaw]:
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
            if CensusPass._targets_second(spec, law) != (spec.hyperedge == HYPEREDGE_LAWS):
                continue
            expansions = HyperedgeTables._expansions(spec, CensusPass._expressions_of(law))
            expanded = CensusPass._substituted(law, expansions) if expansions else law
            if not isinstance(expanded, BaseLaw):
                error_msg = f'The law {law!r} of {spec.owner} is not a law once its reductions are written out.'
                raise TypeError(error_msg)
            kept.append(HyperedgeTables._masked(spec, expanded.prepare(spec.owner)))
        return kept

    @staticmethod
    def _opaque_calls(expression: Expression) -> dict[Expression, Expression]:
        """Return a symbol for every outermost call of a symbolic function.

        Such a call has no value at numbers.
        """
        if isinstance(expression.operator(), SymbolicFunction):
            return {expression: SR.symbol(f'_call_{abs(hash(str(expression)))}')}
        found: dict[Expression, Expression] = {}
        for operand in expression.operands():
            found.update(CensusPass._opaque_calls(operand))
        return found

    @staticmethod
    def _breaks_symmetry(residual: Expression) -> bool:
        """Check whether an expression a symmetry needs to be 0 is not 0.

        It is evaluated at random values of its symbols.
        """
        opaque = residual.subs(CensusPass._opaque_calls(residual))
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
        expected = tensor.matrix_of(value[row, column] for row, column in tensor.indices)
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
    def _component_rates(spec: KernelSpec, law: BaseLaw) -> list[BaseLaw]:
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
    def _role_of(namespace: SimpleNamespace, holder: SimpleNamespace) -> str | None:
        """Return the name in a namespace that holds a namespace or a copy of it.

        It holds it directly or inside, and the result is `None` otherwise.
        """
        for name, value in vars(namespace).items():
            for item in value if isinstance(value, tuple) else (value,):
                if not isinstance(item, SimpleNamespace):
                    continue
                if item is holder or vars(item).keys() == vars(holder).keys() or CensusPass._role_of(item, holder):
                    return name
        return None

    @staticmethod
    def _returned_laws(spec: KernelSpec, component: BasePart) -> Sequence[BasePart]:
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
            return component.prepare().run(spec.namespace).laws_
        except AttributeError as error:
            holder = error.obj
            if not isinstance(component, BaseInteraction) or not isinstance(holder, SimpleNamespace):
                raise
            role = CensusPass._role_of(spec.namespace, holder)
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
    def _declared_laws(spec: KernelSpec, component: BasePart) -> list[BaseLaw]:
        """Return the laws a component returns for its namespace.

        It first checks their classes against its binding.

        Args:
            spec:
                The kernel spec of the component.

            component:
                The object whose `laws` method describes the component.

        Returns:
            laws:
                The laws as declared.

        Raises:
            ValidationError:
                `laws` returns something other than declarable laws, or laws its binding does not take.
        """
        returned = CensusPass._returned_laws(spec, component)
        laws = [law for law in returned if isinstance(law, BaseLaw)]
        if not isinstance(returned, list | tuple) or len(laws) < len(returned):
            error_msg = f'`_get_laws` of {spec.owner} must return a list of laws.'
            raise ValidationError(error_msg)
        split = [item for law in laws for item in CensusPass._component_rates(spec, law)]
        declared = [law.prepare(spec.owner) for law in split]
        if spec.hyperedge is not None:
            declared = CensusPass._hyperedge_laws(spec, declared)
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
            declared = CensusPass._field_laws(spec, [law for law in declared if CensusPass._end_of(spec, law) is None])[
                1
            ]
        if spec.edges is not None:
            edges = spec.edges
            lowered, kept = CensusPass._field_laws(
                edges,
                [law for law in declared if CensusPass._end_of(edges, law) is None],
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
                    CensusPass._end_law(edges, law, end)
                    for law in declared
                    if (end := CensusPass._end_of(edges, law)) is not None
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
    def _component_of(node: NodeSpec) -> BasePart:
        """Return the part whose laws describe a node, a representative member for a group."""
        if isinstance(node.component, GroupElement):
            group = node.component
            return group.element(group.label).set_params(**(group.values or {})).prepare()
        if not isinstance(node.component, BasePart):
            error_msg = f'The node {node.path} holds a {type(node.component).__name__}, which declares no laws.'
            raise TypeError(error_msg)
        return node.component
