"""Define the subsystem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec
from ._local_operator_spec import LocalOperatorSpec
from ._quantum_view_spec import QuantumViewSpec


@dataclass(frozen=True)
class _SubsystemSpec(BaseSpec):
    """Quantum subsystem with its key, its element, its index in the tensor product, its state symbol, and operators."""

    key: str
    element: QuantumViewSpec
    index: int
    symbol: Expression
    operators: Mapping[str, LocalOperatorSpec]
