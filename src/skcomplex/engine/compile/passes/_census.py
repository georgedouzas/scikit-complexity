"""Take the census of the laws every component of a system declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Any, cast

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.elements import BaseElement, GroupElement
from ....core.errors import SkcomplexError
from ....core.interactions import BaseInteraction
from ....core.laws import BaseLaw, NoiseLaw
from ....core.system import System
from ...problems import ElementKind, Entry, VariableKind, components_of, declarations_of, symbol, symbols_of
from .._census import Census, CensusRecord, _declared_laws, _law_owner
from .._compilation import Compilation
from .._declarations import Block, Declarations, _is_interaction
from .._laws import values_of
from .._noise import _process_keys
from .._observables import Observer, _lowerable_observables
from .._pipeline import BasePass
from .._specs import KernelSpec, _component_of, _group_space, _specs_of
from .._tree import Node


class CensusPass(BasePass):
    """Pass that takes the census of the laws of the system, keeping going past failing components."""

    reads = ('nodes',)
    writes = ('census',)

    def _run(self, compilation: Compilation) -> None:
        """Take the census of the laws of the nodes.

        Args:
            compilation:
                The compilation state.
        """
        compilation.census = self._census_of(compilation.system, cast('list[Node]', compilation.nodes))

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
    def _noise_census(spec: KernelSpec, law: object, declarations: Declarations) -> dict[str, Any]:
        """Return the processes, driven states, and read states of every target component of a noise law."""
        if not isinstance(law, NoiseLaw):
            return {}
        targets = components_of(law._get_targets()[0])
        values = (
            components_of(law.value)
            if isinstance(law.value, FreeModuleElement | list | tuple)
            else [law.value] * len(targets)
        )
        references = [spec.references.get(item) for item in targets]
        return {
            'processes': tuple(_process_keys(spec, law, declarations)),
            'driven': tuple((item.entry.name, item.column) for item in references if item is not None),
            'reads': tuple(CensusPass._state_reads(spec, value) for value in values),
        }

    @staticmethod
    def _is_vector(entry: Entry, elements: dict[str, Block]) -> bool:
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
        symbols = symbols_of([value for value in values_of(law, exclude=('process',)) if isinstance(value, Expression)])
        return tuple(sorted(str(item) for item in symbols if item in handles))

    @staticmethod
    def _component_kind(node: Node) -> ElementKind:
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
        block: Block,
        elements: dict[str, Block],
        observe: Observer,
        declarations: Declarations,
        agents: set[str],
    ) -> list[CensusRecord]:
        """Return the census records of the laws of one component."""
        records = []
        for spec in _specs_of(block, elements, declarations.time, observe.evaluate):
            component = cast('BaseElement | BaseInteraction', _component_of(block.node))
            for law in _declared_laws(spec, component):
                items = [item for target in law._get_targets() for item in components_of(target)]
                references = [spec.references.get(item) for item in items if isinstance(item, Expression)]
                targets = tuple(dict.fromkeys(item.entry for item in references if item is not None))
                owner = _law_owner(spec, law, agents)
                vectors = frozenset(entry.name for entry in targets if CensusPass._is_vector(entry, elements))
                kind = CensusPass._component_kind(elements.get(owner, block).node)
                records.append(
                    CensusRecord(
                        block.node.path,
                        owner,
                        law,
                        targets,
                        vectors,
                        kind,
                        _group_space(elements.get(owner, block)),
                        members=spec.rows,
                        handles=CensusPass._handle_uses(law, declarations.handles),
                        **CensusPass._noise_census(spec, law, declarations),
                    ),
                )
        return records

    @staticmethod
    def _census_of(system: System, nodes: list[Node]) -> Census:
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
        declarations = Declarations(time=symbol('t'))
        errors: list[tuple[str, SkcomplexError]] = []
        blocks = []
        for node in nodes:
            try:
                blocks.append(declarations.declare_component(node))
            except SkcomplexError as error:
                errors.append((node.path, error))
        elements = {block.node.path: block for block in blocks if not _is_interaction(block.node)}
        elements.update(declarations.processes)
        observe = Observer(declarations, _lowerable_observables(system, nodes))
        agents = {block.node.path for block in blocks if block.node.is_agent}
        records = []
        for block in blocks:
            try:
                records += CensusPass._records_of(block, elements, observe, declarations, agents)
            except SkcomplexError as error:
                errors.append((block.node.path, error))
        return Census(system, records, errors, declarations.agents)
