"""Define the unknown spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import (
    Expression,
)

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _UnknownSpec(BaseSpec):
    """State component whose equation the analysis writes in its own order, with its function of time."""

    name: str
    function: Expression
    order: int
    indices: tuple[int, ...]
