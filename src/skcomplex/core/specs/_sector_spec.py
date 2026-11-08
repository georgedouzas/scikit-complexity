"""Define the sector spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

from ._base_spec import BaseSpec


@dataclass(frozen=True, eq=False)
class SectorSpec(BaseSpec):
    """Sector of an operator system.

    It holds the basis states whose eigenvalue of a conserved diagonal operator equals a value.

    Args:
        operator:
            A function of the system namespace that returns the conserved operator.

        value:
            The eigenvalue of the operator.

    Examples:
        >>> from skcomplex.core.specs import SectorSpec
        >>> SectorSpec(lambda s: s.n, 2.0).value
        2.0
    """

    operator: Callable[[SimpleNamespace], object]
    value: float
