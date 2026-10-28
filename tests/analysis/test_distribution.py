"""Test histograms, survival functions, Gini coefficients, tail indices, and record counts."""

import numpy as np
import pytest

from skcomplex.analysis import DistributionAnalysis
from skcomplex.core.errors import AnalysisError, ValidationError


def _distribution(kind: str, source: object, **settings: object) -> DistributionAnalysis:
    """Return the run distribution analysis of a source for one kind."""
    return DistributionAnalysis(kind, **settings).check().prepare(source).run()


def test_gini_and_tail_index():
    """Test the Gini coefficient of equal and concentrated values and the Hill tail index."""
    assert _distribution('gini', [5, 5, 5, 5]).gini_ == 0
    assert _distribution('gini', [0, 0, 0, 1]).gini_ == pytest.approx(0.75)
    samples = (1 - np.random.default_rng(3).random(50_000)) ** (-1 / 1.5)
    assert _distribution('tail-index', samples, tail=5000).tail_index_ == pytest.approx(1.5, abs=0.1)
    with pytest.raises(AnalysisError, match='tail = 50000'):
        _distribution('tail-index', samples, tail=50_000)


def test_histogram_and_ccdf():
    """Test that log-binned densities integrate to 1 and the survival function starts at 1."""
    samples = (1 - np.random.default_rng(4).random(1000)) ** (-1 / 1.5)
    histogram = _distribution('histogram', samples, log=True)
    assert np.sum(histogram.density_ * np.diff(histogram.edges_)) == pytest.approx(1.0)
    survival = _distribution('ccdf', samples).survival_
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) < 0)


def test_logarithmic_bins_need_positive_values():
    """Test that logarithmic bins of a sample with zeros raise an error."""
    with pytest.raises(AnalysisError, match='positive values'):
        _distribution('histogram', [0.0, 1.0, 2.0], log=True)


def test_complex_values_are_rejected():
    """Test that complex values raise an error that names the real views."""
    with pytest.raises(ValidationError, match='real part'):
        _distribution('ccdf', np.array([1 + 1j, 2.0]))


def test_record_counts_of_independent_values():
    """Test that 10,000 sequences of 1000 independent values hold the harmonic number of records on average."""
    counts = _distribution('records', np.random.default_rng(1).random((10_000, 1000)).T).records_
    harmonic = np.sum(1 / np.arange(1, 1001))
    assert counts.mean() == pytest.approx(harmonic, rel=0.01)
    assert _distribution('records', np.array([[1, 4], [3, 3], [2, 2], [5, 1]])).records_.tolist() == [3, 1]


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind lists the kinds and a setting the kind ignores is rejected."""
    with pytest.raises(AnalysisError, match='histogram, ccdf'):
        DistributionAnalysis('mean').check()
    with pytest.raises(AnalysisError, match='does not read bins'):
        DistributionAnalysis('gini', bins=10).check()
