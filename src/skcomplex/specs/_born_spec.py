"""Define the born spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec
from ._compiled_flow_spec import _CompiledFlowSpec

if TYPE_CHECKING:
    from ..engine.compile._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _BornSpec(BaseSpec):
    """Transfer of a spec with the amount it moves and its compiled holders."""

    spec: KernelSpec
    flow: _CompiledFlowSpec
    amount: Expression
    sign: int
