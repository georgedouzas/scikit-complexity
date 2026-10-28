"""Build the kernel specs that the laws of every component are lowered with: namespaces, references, and rows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ...core.errors import ValidationError
from ..problems import Problem, Reference
from ..problems._kernels import Search
from ._tree import Node

_MEMBER_REFERENCE = re.compile(r'^(?P<path>.+)\[(?P<index>-?\d+)\]$')
_EDGE_NAMES = ('source', 'target', 'edge')

type Observe = Callable[[Node, object], Observed | None]


@dataclass(frozen=True)
class SetSymbol:
    """Symbol of one element of an element set, with the same symbol of every element of the set."""

    canonical: Expression
    references: tuple[Reference, ...]
    label: str


@dataclass
class KernelSpec:
    """Namespace, references, and evaluation indices that the laws of one component are lowered with."""

    owner: str
    namespace: SimpleNamespace
    references: dict[Expression, Reference]
    rows: int
    pairs: Search | None = None
    roles: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    group: str | None = None
    groups: dict[str, int] = field(default_factory=dict)
    sets: dict[Expression, SetSymbol] = field(default_factory=dict)
    set_index: int = 0
    aligned: tuple[str, ...] = ()

    @staticmethod
    def _target_references(target: object, spec: KernelSpec) -> list[Reference]:
        """Return the references of the state components a law targets."""
        components = Problem._real_components_of(target)
        references = [spec.references.get(component) for component in components]
        if not all(reference is not None and reference.is_state for reference in references):
            error_msg = f'A law of {spec.owner} targets {target!r}, which is not a state symbol of the namespace `s`.'
            raise ValidationError(error_msg)
        return references  # type: ignore[return-value]

    @staticmethod
    def _state_name(spec: KernelSpec, target: Expression) -> str:
        """Return the dotted name of the state a target symbol of a namespace stands for, or the symbol itself."""
        reference = spec.references.get(target)
        return reference.entry.name if reference is not None else str(target)


@dataclass(frozen=True)
class Observed:
    """Value of an observable that a role reads at the current state, with the references of its symbols."""

    namespace: SimpleNamespace
    references: dict[Expression, Reference]
