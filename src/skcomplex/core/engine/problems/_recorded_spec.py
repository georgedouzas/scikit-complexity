"""Define the recorded spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class _RecordedSpec(BaseSpec):
    """Expression a trajectory records at every frame.

    It holds the references of its symbols.
    """

    path: str
    expression: Expression
    references: Mapping[Expression, ReferenceSpec]
