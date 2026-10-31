"""Take the census of the laws every component of a system declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.base import BasePart
from ....core.elements import GroupElement
from ....core.laws import (
    BaseLaw,
    BoundaryLaw,
    CrossingLaw,
    EquationLaw,
    ExtremalLaw,
    JumpLaw,
    LocalRuleLaw,
    MatchLaw,
    NoiseLaw,
    RateLaw,
    RelaxationLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
    UpdateLaw,
)
from ....core.laws.utils import neighborhoods_in, reductions_in
from ....core.specs import CensusRecordSpec, CensusSpec, NodeSpec
from ....core.system import System
from ....errors import SkcomplexError, ValidationError
from ....specs import CallSpec, EntrySpec
from ....types import ElementKind, ProcessKey, VariableKind
from ..._utils import components_of, declarations_of, symbol, symbols_of, vector
from ...problems import BlockSpec, DeclarationsSpec, KernelSpec
from ...problems._edge_search import EdgeSearch
from ...problems._pair_search import PairSearch
from ...problems._topology_kernel import TopologyKernel
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation
from .._observables import ObservableEvaluator
from .._observer import Observer
from .._renewal_law import RenewalLaw
from ._layout import LayoutPass
from ._noise import NoisePass
from ._specs import SpecsPass

EDGE_REQUESTS = ('remove-edge', 'rewire')


class CensusPass(BasePass):
    """Pass that takes the census of the laws of the system, keeping going past failing components."""

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
        """Return the processes, driven states, and read states of every target component of a noise law."""
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
        """Check whether the state of a layout entry is declared as a vector of the space's dimension."""
        path, _, name = entry.name.rpartition('.')
        block = elements.get(path)
        if block is None:
            return False
        declaration = declarations_of(block.node.component_type, VariableKind.STATE).get(name)
        return declaration is not None and declaration.vector

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
                        processes=processes,
                        driven=driven,
                        reads=reads,
                    ),
                )
        return records

    @staticmethod
    def _census_of(system: System, nodes: list[NodeSpec]) -> CensusSpec:
        """Return the census of the laws of a system, with the errors of the components that fail.

        The census declares every component and collects its laws as compilation does, without lowering them, and keeps
        going past a failing component, so validation reports every problem at once.

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
        for node in nodes:
            try:
                declarations, block = LayoutPass._declare_component(declarations, node)
            except SkcomplexError as error:
                errors.append((node.path, error))
                continue
            blocks.append(block)
        elements = {block.node.path: block for block in blocks if not block.node.interaction}
        elements.update(declarations.processes)
        live = ObservableEvaluator._lowerable_observables(system, nodes)
        observer = Observer(declarations, live).check().prepare()
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
        """Return the path of the element or group whose states a renewal law resets, or the law's component."""
        targets = [item for target in (law.reset or {}) for item in components_of(target)]
        reference = spec.references.get(targets[0]) if targets else None
        return spec.owner if reference is None else reference.entry.name.rpartition('.')[0]

    @staticmethod
    def _special_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str | None:
        """Return the owner of a renewal, a rule, a matching, or a topology request, or `None` for another law."""
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
        if isinstance(law, TransferLaw | SampleLaw | EquationLaw | CrossingLaw) or (
            isinstance(law, RelaxationLaw) and not law._get_targets()
        ):
            return spec.owner
        lattice = isinstance(law, RelaxationLaw | LocalRuleLaw | ExtremalLaw)
        target = law._get_targets()[0] if lattice else CensusPass._changed_target(spec, law)
        return spec._target_references(target)[0].entry.name.rpartition('.')[0]

    @staticmethod
    def _expressions_of(law: BaseLaw) -> list[Expression]:
        """Return the symbolic components of the targets and the values of a law, and of the laws it holds."""
        found: list[Expression] = []
        for value in (*law._get_targets(), *law._get_values()):
            if isinstance(value, BaseLaw):
                found += CensusPass._expressions_of(value)
            elif isinstance(value, Expression | FreeModuleElement):
                found += [SR(item) for item in components_of(value)]
        return found

    @staticmethod
    def _outside_reductions(expression: Expression) -> Expression:
        """Return an expression with every reduction and neighbourhood reduction replaced by one placeholder symbol."""
        calls = [item.call for item in reductions_in(expression)] + [item.call for item in neighborhoods_in(expression)]
        if not calls:
            return expression
        placeholder = symbol('_reduction')
        return SR(expression).subs(dict.fromkeys(calls, placeholder))

    @staticmethod
    def _end_of(edges: KernelSpec, law: BaseLaw) -> str | None:
        """Return the end of an edge whose member alone a law reads, or `None` for a law of the edge.

        A law of one end reads no member of the edge group and, outside its reductions, the symbols of one end only,
        and targets states of that end. Inside its reductions, the other end stands for the other members.

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
        """Return a mapping, a sequence, or a call of a law with the symbols of its items replaced."""
        if isinstance(value, Mapping):
            return {
                CensusPass._substituted(key, mapping): CensusPass._substituted(item, mapping)
                for key, item in value.items()
            }
        if isinstance(value, CallSpec):
            return CallSpec(value.function, tuple(CensusPass._substituted(item, mapping) for item in value.arguments))
        return type(value)(CensusPass._substituted(item, mapping) for item in value)

    @staticmethod
    def _substituted(value: object, mapping: Mapping[Expression, Expression]) -> object:
        """Return a value of a law with symbols replaced, through vectors, mappings, sequences, calls, and laws."""
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
        """Return a law of one end of an edge written for the member, with the other end standing for other members.

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
        return substituted.check()

    @staticmethod
    def _declared_laws(spec: KernelSpec, component: BasePart) -> list[BaseLaw]:
        """Return the laws a component returns for its namespace, after checking their classes against its binding.

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
        returned = component.prepare().run(spec.namespace).laws_
        laws = [law for law in returned if isinstance(law, BaseLaw)]
        if not isinstance(returned, list | tuple) or len(laws) < len(returned):
            error_msg = f'`_get_laws` of {spec.owner} must return a list of laws.'
            raise ValidationError(error_msg)
        declared = [law.check().prepare(spec.owner) for law in laws]
        if isinstance(spec.pairs, PairSearch) and not all(isinstance(law, RateLaw) for law in declared):
            error_msg = (
                f'Pair interaction {spec.owner} returns laws other than Rate, which pair interactions do not take.'
            )
            raise ValidationError(error_msg)
        if isinstance(spec.pairs, EdgeSearch):
            declared = [law for law in declared if CensusPass._end_of(spec, law) is None]
        if spec.edges is not None:
            edges = spec.edges
            declared = [
                CensusPass._end_law(edges, law, end)
                for law in declared
                if (end := CensusPass._end_of(edges, law)) is not None
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
        return checked

    @staticmethod
    def _component_of(node: NodeSpec) -> BasePart:
        """Return the part whose laws describe a node, a representative member for a group."""
        if isinstance(node.component, GroupElement):
            group = node.component
            return group.element(group.label).set_params(**(group.values or {})).check()
        if not isinstance(node.component, BasePart):
            error_msg = f'The node {node.path} holds a {type(node.component).__name__}, which declares no laws.'
            raise TypeError(error_msg)
        return node.component
