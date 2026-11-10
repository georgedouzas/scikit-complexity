"""Define the observed spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class ObservedSpec(BaseSpec):
    """A spec of the value of an observable that a role reads at the current state.

    It holds the references of its symbols.
    """

    namespace: SimpleNamespace
    references: dict[Expression, ReferenceSpec]
