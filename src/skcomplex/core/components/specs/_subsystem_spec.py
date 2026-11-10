"""Define the spec of a subsystem of a tensor product."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._local_operator_spec import LocalOperatorSpec
from ._operator_view_spec import OperatorViewSpec


@dataclass(frozen=True)
class SubsystemSpec(BaseSpec):
    """A spec of a subsystem of a tensor product, such as one qubit of a register.

    The compiler of operators builds one per operator element, and the lowering of operators reads them.

    Args:
        key:
            The key of the subsystem, the path of its element.

        element:
            The operator view of the element.

        index:
            The index of the subsystem in the tensor product.

        symbol:
            The state symbol of the subsystem.

        operators:
            The local operators of the element, by name.
    """

    key: str
    element: OperatorViewSpec
    index: int
    symbol: Expression
    operators: Mapping[str, LocalOperatorSpec]
