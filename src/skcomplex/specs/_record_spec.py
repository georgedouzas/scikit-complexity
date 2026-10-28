"""Define the record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec

if TYPE_CHECKING:
    from ..engine.compile._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _RecordSpec(BaseSpec):
    """Recorded expression of a spec, rewritten as later passes substitute hidden states."""

    path: str
    expression: Expression
    spec: KernelSpec
