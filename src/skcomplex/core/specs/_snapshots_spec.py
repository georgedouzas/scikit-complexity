"""Define the snapshots spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class SnapshotsSpec(BaseSpec):
    """A spec of the agents of one group at every frame.

    They are aligned on the agents present in every frame.

    Args:
        times:
            The time of every frame.

        ids:
            The id of every agent, in column order.

        positions:
            The unwrapped positions, frames by agents by dimensions, or `None` without a space with coordinates.

        velocities:
            The velocities, shaped like the positions, or `None` without them.

        directions:
            The direction angles, frames by agents, or `None` without them.

        states:
            Every other state of the agents, frames by agents.

        box:
            The sides of the space, or `None` for an unbounded space.

        periodic:
            Whether opposite sides of the box are joined.

        phases:
            The phase angles, frames by agents, or `None` without them.
    """

    times: Array
    ids: npt.NDArray[np.int64]
    positions: Array | None
    velocities: Array | None
    directions: Array | None
    states: dict[str, npt.NDArray[np.generic]] = field(default_factory=dict)
    box: Array | None = None
    periodic: bool = False
    phases: Array | None = None
