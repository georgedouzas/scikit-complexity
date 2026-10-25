"""Take the census of the laws of a system: the laws every component declares, with their owners and targets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...core.base import (
    BaseLaw,
    BaseSpace,
    ComponentKind,
    Entry,
    Expression,
    Registry,
    SkcomplexError,
    SymbolicVector,
    Trait,
    UnregisteredLawError,
    ValidationError,
    VariableKind,
    components_of,
    declarations_of,
    has_trait,
    is_registered,
    law_classes,
    symbol,
    symbols_of,
)
from ...core.components import GroupComponent
from ...core.laws import BoundaryLaw, EquationLaw, EventLaw, JumpLaw, NoiseLaw, Objective, RateLaw, UpdateLaw, values_of
from ...core.problems import EdgeSearch
from ...core.problems._kernels import PairSearch
from ...core.system import System
from ._declarations import Block, Declarations, is_interaction
from ._noise import ProcessKey, process_keys
from ._observables import Observer, lowerable_observables
from ._specs import KernelSpec, component_of, group_space, specs_of, target_references
from ._tree import Node

type OwnerRule = Callable[[KernelSpec, BaseLaw, set[str]], str | None]

OWNER_RULES = Registry[OwnerRule]('owner rule')


def _check_registered(spec: KernelSpec, declared: object) -> None:
    """Raise an error for a law of a class that neither the shared core nor a formalism registered."""
    laws = declared if isinstance(declared, list | tuple) else []
    for law in laws:
        if isinstance(law, BaseLaw) and not is_registered(law):
            error_msg = (
                f'{spec.owner} returns a {type(law).__name__}, a law class that no formalism registered. A '
                f'formalism must register {type(law).__name__} with `register(Formalism(..., laws=(...)))` before a '
                'system holds its laws.'
            )
            raise UnregisteredLawError(error_msg)


def _changed_target(spec: KernelSpec, law: BaseLaw) -> object:
    """Return the target of a law, or the first state a jump changes."""
    if isinstance(law, JumpLaw):
        return next(iter(law.changes))
    if not law.targets:
        error_msg = f'A {type(law).__name__} of {spec.owner} changes no state of its component.'
        raise ValidationError(error_msg)
    return law.targets[0]


def _handle_uses(law: BaseLaw, handles: Mapping[Expression, tuple[str, int]]) -> tuple[str, ...]:
    """Return the Wiener handles that a law uses outside the process of a noise law."""
    if not handles:
        return ()
    symbols = symbols_of([value for value in values_of(law, exclude=('process',)) if isinstance(value, Expression)])
    return tuple(sorted(str(item) for item in symbols if item in handles))


def _state_reads(spec: KernelSpec, value: object) -> frozenset[tuple[str, int]]:
    """Return the state components that a law value depends on."""
    if not isinstance(value, Expression):
        return frozenset()
    return frozenset(
        (reference.entry.name, reference.column)
        for item in value.variables()
        if (reference := spec.references.get(item)) is not None and reference.is_state
    )


def _noise_census(spec: KernelSpec, law: object, declarations: Declarations) -> dict[str, Any]:
    """Return the processes, driven states, and read states of every target component of a noise law."""
    if not isinstance(law, NoiseLaw):
        return {}
    targets = components_of(law.targets[0])
    values = (
        components_of(law.value) if isinstance(law.value, SymbolicVector | list | tuple) else [law.value] * len(targets)
    )
    references = [spec.references.get(item) for item in targets]
    return {
        'processes': tuple(process_keys(spec, law, declarations)),
        'driven': tuple((item.entry.name, item.column) for item in references if item is not None),
        'reads': tuple(_state_reads(spec, value) for value in values),
    }


def _component_kind(node: Node) -> ComponentKind:
    """Return the kind of the component of a node."""
    if isinstance(node.component, GroupComponent) and node.component.variable:
        return ComponentKind.VARIABLE_GROUP
    if isinstance(node.component, GroupComponent):
        parts = {'sites': ComponentKind.SITE_GROUP, 'edges': ComponentKind.EDGE_GROUP}
        parts['hyperedges'] = ComponentKind.HYPEREDGE_GROUP
        return parts.get(node.component.on or '', node.component.component_kind)
    return ComponentKind.ELEMENT


def _is_vector(entry: Entry, elements: dict[str, Block]) -> bool:
    """Check whether the state of a layout entry is declared as a vector of the space's dimension."""
    path, _, name = entry.name.rpartition('.')
    block = elements.get(path)
    if block is None:
        return False
    declaration = declarations_of(block.node.component_type, VariableKind.STATE).get(name)
    return declaration is not None and declaration.vector


def _records_of(
    block: Block,
    elements: dict[str, Block],
    observe: Observer,
    declarations: Declarations,
    agents: set[str],
) -> list[CensusRecord]:
    """Return the census records of the laws of one component."""
    records = []
    for spec in specs_of(block, elements, declarations.time, observe.evaluate):
        for law in declared_laws(spec, component_of(block.node)):
            items = [item for target in law.targets for item in components_of(target)]
            references = [spec.references.get(item) for item in items if isinstance(item, Expression)]
            targets = tuple(dict.fromkeys(item.entry for item in references if item is not None))
            owner = law_owner(spec, law, agents)
            vectors = frozenset(entry.name for entry in targets if _is_vector(entry, elements))
            kind = _component_kind(elements.get(owner, block).node)
            records.append(
                CensusRecord(
                    block.node.path,
                    owner,
                    law,
                    targets,
                    vectors,
                    kind,
                    group_space(elements.get(owner, block)),
                    members=spec.rows,
                    handles=_handle_uses(law, declarations.handles),
                    **_noise_census(spec, law, declarations),
                ),
            )
    return records


def register_owner(rule: OwnerRule) -> None:
    """Register a rule that names the owner of the laws it knows, keeping it once.

    Args:
        rule:
            The function that takes a kernel spec, a law, and the paths of the agents, and returns the path of the
            element or group the law belongs to, or `None` for a law it does not know.
    """
    OWNER_RULES.register(rule, rule)


def law_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str:
    """Return the path of the element or group whose state a law changes, or that a registered rule names.

    Equations, events, and objectives belong to their component.

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
    for key in OWNER_RULES.keys:
        owner = OWNER_RULES.resolve(key)(spec, law, agents)
        if owner is not None:
            return owner
    if isinstance(law, EquationLaw | EventLaw | Objective):
        return spec.owner
    return target_references(_changed_target(spec, law), spec)[0].entry.name.rpartition('.')[0]


def kind_of(law: BaseLaw) -> type:
    """Return the class of a law that decides its form, with every noise law counted as a noise law.

    Args:
        law:
            The law.

    Returns:
        kind:
            The class.
    """
    return NoiseLaw if isinstance(law, NoiseLaw) else type(law)


def declared_laws(spec: KernelSpec, component: object) -> list[BaseLaw]:
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
        UnregisteredLawError:
            A law is of a class that no formalism registered.

        ValidationError:
            `laws` returns something other than declarable laws, or laws its binding does not take.
    """
    declared: Any = component.run(spec.namespace)  # type: ignore[attr-defined]
    _check_registered(spec, declared)
    if not isinstance(declared, list | tuple) or not all(has_trait(law, Trait.DECLARABLE) for law in declared):
        names = [law.__name__ for law in law_classes(Trait.DECLARABLE)]
        error_msg = f'`laws` of {spec.owner} must return a list of {", ".join(names[:-1])}, or {names[-1]} laws.'
        raise ValidationError(error_msg)
    if isinstance(spec.pairs, PairSearch) and not all(isinstance(law, RateLaw) for law in declared):
        error_msg = f'Pair interaction {spec.owner} returns laws other than Rate, which pair interactions do not take.'
        raise ValidationError(error_msg)
    edge_kinds = (RateLaw, NoiseLaw, JumpLaw, UpdateLaw, BoundaryLaw)
    if isinstance(spec.pairs, EdgeSearch) and not all(isinstance(law, edge_kinds) for law in declared):
        error_msg = (
            f'Interaction {spec.owner} over neighbours returns laws other than Rate, Noise, Jump, Update, or Boundary '
            'laws, which its edges do not take.'
        )
        raise ValidationError(error_msg)
    checked = []
    for law in declared:
        try:
            checked.append(law.check())
        except ValidationError as error:
            error_msg = f'{type(law).__name__} of {spec.owner}: {error}'
            raise ValidationError(error_msg) from error
    return checked


def census_of(system: System, nodes: list[Node]) -> Census:
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
    elements = {block.node.path: block for block in blocks if not is_interaction(block.node)}
    elements.update(declarations.processes)
    observe = Observer(declarations, lowerable_observables(system, nodes))
    agents = {block.node.path for block in blocks if block.node.is_agent}
    records = []
    for block in blocks:
        try:
            records += _records_of(block, elements, observe, declarations, agents)
        except SkcomplexError as error:
            errors.append((block.node.path, error))
    return Census(system, records, errors)


@dataclass(frozen=True)
class CensusRecord:
    """Law of a component with the element it changes and the states it targets.

    Args:
        path:
            The path of the component that returns the law.

        owner:
            The path of the element or group whose state the law changes, or that a rule acts for.

        law:
            The law.

        targets:
            The layout entries of the states the law targets.

        vectors:
            The names of the targets declared as vector states.

        component:
            The kind of the owner: an element, a fixed or variable group, or a group on the sites of a space.

        space:
            The nearest space enclosing the component.

        processes:
            For a noise law, the Wiener process of every target component: `None` for an independent process,
            `('local', path, label)` for a local label, or `('wiener', path, index)` for a process of a `WienerProcess`.

        driven:
            For a noise law, the state entry name and column of every target component.

        reads:
            For a noise law, the state components that the amplitude of every target component depends on.

        members:
            The number of members the law is evaluated for.

        handles:
            The handles of Wiener processes that the law uses outside the process of a noise law.
    """

    path: str
    owner: str
    law: BaseLaw
    targets: tuple[Entry, ...]
    vectors: frozenset[str]
    component: ComponentKind
    space: BaseSpace | None
    processes: tuple[ProcessKey | None, ...] = ()
    driven: tuple[tuple[str, int], ...] = ()
    reads: tuple[frozenset[tuple[str, int]], ...] = ()
    members: int = 1
    handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class Census:
    """Laws of every component of a system, and the errors of the components that fail to declare or collect.

    Args:
        system:
            The system.

        records:
            The laws of the components that declare and collect without error.

        errors:
            The path of every failing component with its error.
    """

    system: System
    records: list[CensusRecord]
    errors: list[tuple[str, SkcomplexError]]
