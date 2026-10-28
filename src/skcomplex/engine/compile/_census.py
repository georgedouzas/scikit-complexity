"""Take the census of the laws of a system: the laws every component declares, with their owners and targets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...core.errors import SkcomplexError
from ...core.laws import BaseLaw
from ...core.spaces import BaseSpace
from ...core.system import System
from ..problems import ElementKind, Entry
from ._noise import ProcessKey


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
