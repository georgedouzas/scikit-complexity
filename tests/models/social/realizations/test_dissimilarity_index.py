"""Test the `_dissimilarity_index` module of `skcomplex.models.social.realizations`."""

import numpy as np

from skcomplex.analysis import DistributionAnalysis
from skcomplex.models.social.realizations import DissimilarityIndexAnalysis


def test_dissimilarity_index_equals_the_total_variation():
    """Test that the dissimilarity index equals the total variation of two groups across units."""
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.integers(0, 2, 200), rng.integers(0, 7, 200)])
    index = DissimilarityIndexAnalysis().check().prepare(rows).run().distance_
    assert index == DistributionAnalysis('total-variation').check().prepare(rows).run().distance_
    separated = np.column_stack([[0, 0, 1, 1], [0, 0, 1, 1]])
    assert DissimilarityIndexAnalysis().check().prepare(separated).run().distance_ == 1
