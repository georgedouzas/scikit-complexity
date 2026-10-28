"""Test the dissimilarity index of groups across units."""

import pytest

from skcomplex.analysis import dissimilarity_index
from skcomplex.core.errors import AnalysisError


def test_dissimilarity_index_of_separated_and_mixed_groups():
    """Test that separated groups give 1, even mixing 0, and more than two groups an error."""
    assert dissimilarity_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert dissimilarity_index([0, 1, 0, 1], [0, 0, 1, 1]) == 0
    with pytest.raises(AnalysisError, match='two groups'):
        dissimilarity_index([0, 1, 2], [0, 0, 1])
