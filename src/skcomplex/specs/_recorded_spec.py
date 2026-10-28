"""Define the recorded spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _RecordedSpec(BaseSpec):
    """Expression a trajectory records at every frame, with the references of its symbols."""

    path: str
    expression: Expression
    references: Mapping[Expression, Any]
