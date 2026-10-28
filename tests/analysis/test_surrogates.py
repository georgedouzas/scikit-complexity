"""Test surrogate tests of measures of dependence."""

import numpy as np
import pytest

from skcomplex.analysis import InformationAnalysis, SurrogateTestAnalysis
from skcomplex.errors import AnalysisError


def _test(kind: str, data: np.ndarray, **settings: object) -> SurrogateTestAnalysis:
    """Return the surrogate test of an information measure of the columns of data, run."""
    analysis = SurrogateTestAnalysis(InformationAnalysis(kind), 'information_', **settings)
    return analysis.check().prepare(data).run()


def test_surrogate_test_independent_variables_are_not_significant():
    """Test that independent bits give a large p-value for their total correlation."""
    data = np.random.default_rng(0).integers(0, 2, (2000, 3))
    test = _test('total-correlation', data, n=49, alternative='greater', random_state=1)
    assert test.p_value_ > 0.05
    assert test.surrogates_.shape == (49,)


def test_surrogate_test_synergy_is_significant():
    """Test that the negative O-information of XOR lies below every surrogate."""
    bits = np.random.default_rng(2).integers(0, 2, (2000, 2))
    data = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
    test = _test('o-information', data, n=19, alternative='less', random_state=3)
    assert test.value_ == pytest.approx(-1.0, abs=0.01)
    assert test.p_value_ == pytest.approx(0.05)


def test_surrogate_test_rejects_invalid_settings():
    """Test that an unknown alternative, no surrogates, and a measure that is not an analysis raise errors."""
    with pytest.raises(AnalysisError, match='two-sided'):
        SurrogateTestAnalysis(InformationAnalysis('total-correlation'), 'information_', alternative='both').check()
    with pytest.raises(AnalysisError, match='n = 0'):
        SurrogateTestAnalysis(InformationAnalysis('total-correlation'), 'information_', n=0).check()
    with pytest.raises(AnalysisError, match='is an analysis'):
        SurrogateTestAnalysis(np.mean, 'information_').check()
