"""Test surrogate tests of measures of dependence."""

import numpy as np
import pytest

from skcomplex.core.analysis import o_information, surrogate_test, total_correlation
from skcomplex.core.base import AnalysisError


def test_surrogate_test_independent_variables_are_not_significant():
    """Test that independent bits give a large p-value for their total correlation."""
    data = np.random.default_rng(0).integers(0, 2, (2000, 3))
    test = surrogate_test(total_correlation, data, n=49, alternative='greater', random_state=1)
    assert test.p_value > 0.05
    assert test.surrogates.shape == (49,)


def test_surrogate_test_synergy_is_significant():
    """Test that the negative O-information of XOR lies below every surrogate."""
    bits = np.random.default_rng(2).integers(0, 2, (2000, 2))
    data = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
    test = surrogate_test(o_information, data, n=19, alternative='less', random_state=3)
    assert test.value == pytest.approx(-1.0, abs=0.01)
    assert test.p_value == pytest.approx(0.05)


def test_surrogate_test_unknown_alternative():
    """Test that an unknown alternative raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='two-sided'):
        surrogate_test(total_correlation, np.zeros((10, 2)), alternative='both')
