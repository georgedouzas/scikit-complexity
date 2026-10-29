"""Define the request spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._draw_spec import DrawSpec
from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _RequestSpec(BaseSpec):
    """DrawSpec of a component that needs a value per evaluation row at every step."""

    owner: str
    name: str
    normal: bool
    rows: int
    slot: Expression
    spec: KernelSpec
    draw: DrawSpec | None = None
    slots: Mapping[str, Expression] = field(default_factory=dict)
