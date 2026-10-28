"""Name the milling order of a swarm, the alignment of the rotation of its agents about their centre."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....analysis import AlignmentAnalysis


class MillingOrderAnalysis(AlignmentAnalysis):
    """Milling order of a swarm at every frame, the mean normalized angular momentum about its centre.

    It is the alignment of kind `'mean-rotation'`: 1 when every agent turns the same way about the centre, as in a
    fish mill, and near 0 for a polarized or disordered group.

    Args:
        group:
            The path of the agent group of a trajectory, or `None` for the first one.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.physics.active_matter import MillingOrderAnalysis
        >>> angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        >>> ring = np.column_stack([np.cos(angles), np.sin(angles)])
        >>> mill = np.hstack([ring, np.column_stack([-ring[:, 1], ring[:, 0]])])
        >>> MillingOrderAnalysis().check().prepare(mill).run().order_.round(6).tolist()
        [1.0]
    """

    def __init__(self, group: str | None = None) -> None:
        super().__init__('mean-rotation', group=group)
