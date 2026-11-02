"""Define the subsystem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._local_operator_spec import LocalOperatorSpec
from ._operator_view_spec import OperatorViewSpec


@dataclass(frozen=True)
class _SubsystemSpec(BaseSpec):
    """Subsystem with its key, its element, its index in the tensor product, its state symbol, and operators."""

    key: str
    element: OperatorViewSpec
    index: int
    symbol: Expression
    operators: Mapping[str, LocalOperatorSpec]
