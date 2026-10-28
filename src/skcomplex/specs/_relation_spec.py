"""Define the relation spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec

if TYPE_CHECKING:
    from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _RelationSpec(BaseSpec):
    """Equation of a component as `expression == 0`, with the placeholders of its highest derivatives."""

    spec: KernelSpec
    expression: Expression
    highest: dict[Expression, Expression]
    description: str
