"""Define the jump channels spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _JumpChannelsSpec(BaseSpec):
    """Random events of one jump law, one channel per evaluation index."""

    offset: int
    targets: npt.NDArray[np.intp]
    increments: Array
    changes: tuple[Callable[[float, Array, Array], Array] | None, ...] = ()

    def get_increments_at(self, t: float, x: Array, p: Array, member: int) -> Array:
        """Return the increment of every target at a firing.

        Only a target that has a function gets an increment from it.
        """
        if not self.changes:
            return self.increments
        return np.array(
            [
                increment if change is None else change(t, x, p)[member]
                for increment, change in zip(self.increments, self.changes, strict=True)
            ],
            dtype=float,
        )

    @property
    def size(self) -> int:
        """Number of channels."""
        return int(self.targets.shape[1])
