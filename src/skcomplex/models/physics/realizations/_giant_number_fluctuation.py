"""Define the number fluctuations of active matter, giant in ordered flocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import ScalingAnalysis


class GiantNumberFluctuationAnalysis(ScalingAnalysis):
    """Number fluctuations of particles in boxes of growing side, giant in ordered active matter.

    It is the scaling analysis of kind `'variance-mean'` of the box counts of a density field, the slope of the log
    standard deviation of the number against its log mean. Independent particles give 1/2, and ordered flocks give
    giant fluctuations above it, near 0.8 for the Vicsek model.

    Args:
        variables:
            The lattice state of a trajectory to read, or `None` for the first one.

        scales:
            The box sides in cells, or `None` for powers of 2 up to a quarter of the smallest side.

    Attributes:
        values_:
            The density frames.

        labels_:
            The label of the field.

        scales_:
            The box sides.

        means_:
            The mean number in boxes of every side.

        deviations_:
            The standard deviation of the number in boxes of every side.

        exponent_:
            The slope of the log deviation against the log mean.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.physics.realizations import GiantNumberFluctuationAnalysis
        >>> counts = np.random.default_rng(0).poisson(5.0, (20, 64, 64))
        >>> round(GiantNumberFluctuationAnalysis().check().prepare(counts).run().exponent_, 1)
        0.5
    """

    def __init__(self, variables: Sequence[str] | None = None, *, scales: Sequence[int] | None = None) -> None:
        super().__init__("variance-mean", variables, scales=scales)
