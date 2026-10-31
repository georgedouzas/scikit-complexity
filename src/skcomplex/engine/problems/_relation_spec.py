"""Define the relation spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _RelationSpec(BaseSpec):
    """Equation of a component as `expression == 0`, with the placeholders of its highest derivatives.

    The key names the state whose equation it is, as its entry and column, so that the relations of one key add, or
    is `None` for a relation of its own.
    """

    spec: KernelSpec
    expression: Expression
    highest: dict[Expression, Expression]
    description: str
    key: tuple[str, int] | None = None
