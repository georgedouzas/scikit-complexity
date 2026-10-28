"""Random events of one jump law, one channel per evaluation index."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...core.types import Vector
from ._base import BaseValue


@dataclass(frozen=True)
class _JumpChannels(BaseValue):
    """Random events of one jump law, one channel per evaluation index."""

    offset: int
    targets: npt.NDArray[np.intp]
    increments: Vector

    @property
    def size(self) -> int:
        """Number of channels."""
        return int(self.targets.shape[1])
