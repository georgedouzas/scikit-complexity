"""Define the jump channels spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

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

    @property
    def size(self) -> int:
        """Number of channels."""
        return int(self.targets.shape[1])
