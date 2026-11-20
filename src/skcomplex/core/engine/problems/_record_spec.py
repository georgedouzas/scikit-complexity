"""Define the record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _RecordSpec(BaseSpec):
    """Recorded expression of a spec.

    It is rewritten as later passes substitute hidden states.
    """

    path: str
    expression: Expression
    spec: KernelSpec
