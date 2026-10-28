"""Define Duncan's dissimilarity index of two groups across units."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import DistributionAnalysis


class DissimilarityIndexAnalysis(DistributionAnalysis):
    """Duncan's dissimilarity index of two groups across units, the share of one that would have to move.

    It is the distribution analysis of kind `'total-variation'`, `1/2 Σ_u |a_u / A - b_u / B|` from one row per agent
    with its group and its unit, such as a district or a block of cells, 0 for even mixing and 1 for complete
    segregation.

    Args:
        variables:
            The group and the unit series of a trajectory to read, or `None` for every scalar series.

    Attributes:
        values_:
            The group and the unit of every agent.

        labels_:
            The labels of the two columns.

        distance_:
            The dissimilarity index.

    Examples:
        >>> from skcomplex.models.social.realizations import DissimilarityIndexAnalysis
        >>> DissimilarityIndexAnalysis().check().prepare([[0, 0], [0, 0], [1, 1], [1, 1]]).run().distance_
        1.0
    """

    def __init__(self, variables: Sequence[str] | None = None) -> None:
        super().__init__('total-variation', variables)
