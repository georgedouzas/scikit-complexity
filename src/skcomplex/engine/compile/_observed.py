"""Value of an observable that a role reads at the current state, with the references of its symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ..problems import Reference


@dataclass(frozen=True)
class Observed:
    """Value of an observable that a role reads at the current state, with the references of its symbols."""

    namespace: SimpleNamespace
    references: dict[Expression, Reference]
