"""Take the census of the laws of a system: the laws every component declares, with their owners and targets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sage.all__sagemath_symbolics import Expression

from ...core.errors import SkcomplexError, ValidationError
from ...core.laws import (
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
from ...core.spaces import BaseSpace
from ...core.system import System
from ..problems import EdgeSearch, ElementKind, Entry, components_of
from ..problems._kernels import PairSearch
from ._noise import ProcessKey
from ._renewal_law import RenewalLaw
from ._specs import KernelSpec, target_references
from ._topology import _request_of


def _changed_target(spec: KernelSpec, law: BaseLaw) -> object:
    """Return the target of a law, or the first state a jump changes."""
    if isinstance(law, JumpLaw):
        return next(iter(law.changes))
    if not law._get_targets():
        error_msg = f'A {type(law).__name__} of {spec.owner} changes no state of its component.'
        raise ValidationError(error_msg)
    return law._get_targets()[0]


def _renewal_group(spec: KernelSpec, law: RenewalLaw) -> str:
    """Return the path of the element or group whose states a renewal law resets, or the law's component."""
    targets = [item for target in (law.reset or {}) for item in components_of(target)]
    reference = spec.references.get(targets[0]) if targets else None
    return spec.owner if reference is None else reference.entry.name.rpartition('.')[0]


def _special_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str | None:
    """Return the owner of a renewal, a rule, a matching, or a topology request, or `None` for another law."""
    if isinstance(law, JumpLaw) and law.waiting is not None:
        return _renewal_group(spec, RenewalLaw(law.waiting, law.reset))
    if isinstance(law, RenewalLaw):
        return _renewal_group(spec, law)
    if isinstance(law, RuleLaw | MatchLaw):
        paths = [path for value in spec.roles.values() for path in ([value] if isinstance(value, str) else value)]
        return next((path for path in paths if path in agents), next(iter(paths), spec.owner))
    if _request_of(law) is not None:
        return spec.group or next(iter(spec.groups), spec.owner)
    return None


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
    owner = _special_owner(spec, law, agents)
    if owner is not None:
        return owner
    if isinstance(law, TransferLaw | SampleLaw | EquationLaw | CrossingLaw) or (
        isinstance(law, RelaxationLaw) and not law._get_targets()
    ):
        return spec.owner
    lattice = isinstance(law, RelaxationLaw | LocalRuleLaw | ExtremalLaw)
    target = law._get_targets()[0] if lattice else _changed_target(spec, law)
    return target_references(target, spec)[0].entry.name.rpartition('.')[0]


def _kind_of(law: BaseLaw) -> type:
    """Return the class of a law that decides its form, with every noise law counted as a noise law.

    Args:
        law:
            The law.

    Returns:
        kind:
            The class.
    """
    return NoiseLaw if isinstance(law, NoiseLaw) else type(law)


def _declared_laws(spec: KernelSpec, component: object) -> list[BaseLaw]:
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
    declared: Any = component.run(spec.namespace)  # type: ignore[attr-defined]
    if not isinstance(declared, list | tuple) or not all(isinstance(law, BaseLaw) for law in declared):
        error_msg = f'`_get_laws` of {spec.owner} must return a list of laws.'
        raise ValidationError(error_msg)
    declared = [law.check().prepare(spec.owner) for law in declared]
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
            checked.append(law.prepare(spec.owner))
        except ValidationError as error:
            error_msg = f'{type(law).__name__} of {spec.owner}: {error}'
            raise ValidationError(error_msg) from error
    return checked


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
            `('local', path, label)` for a local label, or `('wiener', path, index)` for a process of a
            `WienerProcessSpec`.

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
    component: ElementKind
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

        agents:
            The path of the agent group and the variable of every symbol that reduces agents, an empty variable for
            the symbol of the members.
    """

    system: System
    records: list[CensusRecord]
    errors: list[tuple[str, SkcomplexError]]
    agents: dict[Expression, tuple[str, str]] = field(default_factory=dict)
