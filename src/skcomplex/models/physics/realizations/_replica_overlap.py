"""Define the overlap of two replicas of a spin configuration."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import CorrelationAnalysis


class ReplicaOverlapAnalysis(CorrelationAnalysis):
    """Overlap `q = <s_i t_i>` of two replicas of a spin configuration, the order parameter of spin glasses.

    It is the correlation analysis of kind `'overlap'`. Replicas that share a frozen state overlap near ±1, and
    independent paramagnetic replicas near 0.

    Args:
        variables:
            The lattice state of the trajectories to read, or `None` for the first one.

    Attributes:
        values_:
            The two replicas.

        labels_:
            The labels of the two replicas.

        overlap_:
            The overlap, one per frame for stacks of frames.

    Examples:
        >>> from skcomplex.models.physics.realizations import ReplicaOverlapAnalysis
        >>> ReplicaOverlapAnalysis().check().prepare([[1, -1, 1, 1], [1, -1, -1, 1]]).run().overlap_
        0.5
    """

    def __init__(self, variables: Sequence[str] | None = None) -> None:
        super().__init__("overlap", variables)
