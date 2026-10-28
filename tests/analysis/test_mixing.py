"""Test the mixing analysis, the dissimilarity index of groups across units."""

import numpy as np
import pytest

from skcomplex.analysis import MixingAnalysis
from skcomplex.core.errors import AnalysisError


def _dissimilarity(groups: list[int], units: list[int]) -> float:
    """Return the dissimilarity index of the groups across the units."""
    return MixingAnalysis().check().prepare(np.column_stack([groups, units])).run().dissimilarity_


def test_dissimilarity_index_of_separated_and_mixed_groups():
    """Test that separated groups give 1, even mixing 0, and more than two groups an error."""
    assert _dissimilarity([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert _dissimilarity([0, 1, 0, 1], [0, 0, 1, 1]) == 0
    with pytest.raises(AnalysisError, match='two groups'):
        _dissimilarity([0, 1, 2], [0, 0, 1])


def test_mixing_rejects_a_source_without_units():
    """Test that a source without one group and one unit per agent raises an error."""
    with pytest.raises(AnalysisError, match='one row per agent'):
        MixingAnalysis().check().prepare([0, 1])
