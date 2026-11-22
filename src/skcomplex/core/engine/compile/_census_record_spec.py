"""Define the spec of one law of a component, with what it changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...base import BaseSpec
from ...components import EntrySpec
from ...components.laws import BaseLaw
from ...components.spaces import BaseSpace
from ...enums import ElementKind
from ...types import ProcessKey


@dataclass(frozen=True)
class _CensusRecordSpec(BaseSpec):
    """A spec of one law of a component, with the element and the states it changes.

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

        degrees:
            The degree of the cells of every symbol the law reads of a group on a cell complex, by the name of the
            symbol.

        differentiated:
            For an equation, the paths of the elements and groups whose states it reads a derivative of, such as the
            cart of `(M + m) x'' + m l theta'' cos(theta) == F`.
    """

    path: str
    owner: str
    law: BaseLaw
    targets: tuple[EntrySpec, ...]
    vectors: frozenset[str]
    component: ElementKind
    space: BaseSpace | None
    processes: tuple[ProcessKey | None, ...] = ()
    driven: tuple[tuple[str, int], ...] = ()
    reads: tuple[frozenset[tuple[str, int]], ...] = ()
    members: int = 1
    handles: tuple[str, ...] = ()
    degrees: Mapping[str, int] = field(default_factory=dict, compare=False)
    differentiated: frozenset[str] = frozenset()
