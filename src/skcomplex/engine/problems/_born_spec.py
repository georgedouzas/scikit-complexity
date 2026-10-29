"""Define the born spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._compiled_flow_spec import _CompiledFlowSpec
from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _BornSpec(BaseSpec):
    """Transfer of a spec with the amount it moves and its compiled holders."""

    spec: KernelSpec
    flow: _CompiledFlowSpec
    amount: Expression
    sign: int
