"""Define the request spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec
from ._draw_spec import DrawSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class _RequestSpec(BaseSpec):
    """DrawSpec of a component that needs a value per evaluation row at every step."""

    owner: str
    name: str
    normal: bool
    rows: int
    slot: Expression
    references: dict[Expression, ReferenceSpec]
    draw: DrawSpec | None = None
    slots: dict[str, Expression] = field(default_factory=dict)
