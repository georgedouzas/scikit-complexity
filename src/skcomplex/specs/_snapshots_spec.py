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
    """Agents of one group at every frame, aligned on the agents present in every frame.

    Args:
        times:
            The time of every frame.

        ids:
            The id of every agent, in column order.

        positions:
            The unwrapped positions, frames by agents by dimensions, or `None` without a space with coordinates.

        velocities:
            The velocities, shaped like the positions, or `None` without them.

        headings:
            The heading angles, frames by agents, or `None` without them.

        states:
            Every other state of the agents, frames by agents.

        box:
            The sides of the space, or `None` for an unbounded space.

        periodic:
            Whether opposite sides of the box are joined.
    """

    times: Array
    ids: npt.NDArray[np.int64]
    positions: Array | None
    velocities: Array | None
    headings: Array | None
    states: dict[str, npt.NDArray[np.generic]] = field(default_factory=dict)
    box: Array | None = None
    periodic: bool = False
