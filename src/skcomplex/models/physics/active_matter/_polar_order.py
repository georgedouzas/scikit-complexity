"""Name the polar order of a flock, the alignment of its headings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import AlignmentAnalysis


class PolarOrderAnalysis(AlignmentAnalysis):
    """Polar order of a flock at every frame, the length of the mean heading.

    It is the alignment of kind `'mean-vector'`: 1 when every agent moves the same way and about `1 / sqrt(n)` for
    random headings, the order parameter of the Vicsek model.

    Args:
        variables:
            The field or the series of a trajectory to read, or `None` for its agents.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        vectors:
            Whether the members hold velocity vectors rather than heading angles.

    Examples:
        >>> from skcomplex.models.physics.active_matter import PolarOrderAnalysis
        >>> PolarOrderAnalysis().check().prepare([0.0, 0.0, 0.0]).run().order_.tolist()
        [1.0]
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        group: str | None = None,
        vectors: bool = False,
    ) -> None:
        super().__init__('mean-vector', variables, group=group, vectors=vectors)
