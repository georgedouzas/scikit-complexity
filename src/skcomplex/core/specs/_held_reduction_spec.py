"""Define the held reduction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..types import AnyArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _HeldReductionSpec(BaseSpec):
    """Reduction of a group that a co-simulation holds between synchronisation points.

    The co-simulation computes it at every synchronisation point and holds it until the next.
    """

    group: str
    reduce: str | Callable[[AnyArray], object]
    variables: tuple[str, ...] = ()
    symbols: tuple[Expression, ...] = ()
    argument: Expression | None = None
    condition: Expression | None = None

    @property
    def key(self) -> tuple[str, ...]:
        """Group, reduction, argument, and condition.

        They identify the reduction.
        """
        return (self.group, str(self.reduce), str(self.argument), str(self.condition))
