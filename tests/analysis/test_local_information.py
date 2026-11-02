"""Test that local information measures average to the global measures."""

import numpy as np
import pytest

from skcomplex.analysis import EntropyAnalysis, InformationAnalysis, LocalInformationAnalysis
from skcomplex.errors import AnalysisError


def _local(kind: str, source: object, **settings: object) -> np.ndarray:
    """Return the local information of a source for one kind."""
    return LocalInformationAnalysis(kind, **settings).check().prepare(source).run().information_


def _global(kind: str, source: object, **settings: object) -> np.ndarray | float:
    """Return the global information of a source for one kind."""
    return InformationAnalysis(kind, **settings).check().prepare(source).run().information_


def test_local_mutual_information_gaussian_mean_is_global():
    """Test that the mean local mutual information of correlated Gaussians equals the global one within 1e-9."""
    rng = np.random.default_rng(0)
    data = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 1.0]], 50_000)
    local = _local('mutual', data, estimator='gaussian')
    assert local.shape == (2, 2, 50_000)
    assert np.isnan(local[0, 0]).all()
    expected = _global('total-correlation', data, estimator='gaussian')
    assert local[0, 1].mean() == pytest.approx(expected, abs=1e-9)


def test_local_mutual_information_plugin_mean_is_global():
    """Test that the mean local mutual information of symbols equals their plug-in mutual information."""
    rng = np.random.default_rng(1)
    first = rng.integers(0, 3, 5000)
    second = (first + rng.integers(0, 2, 5000)) % 3
    data = np.column_stack([first, second])
    assert _local('mutual', data)[0, 1].mean() == pytest.approx(_global('mutual', data)[0, 1], abs=1e-9)


def test_local_entropy_mean_is_shannon_entropy():
    """Test that the mean local entropy of symbols equals their Shannon entropy."""
    symbols = np.random.default_rng(2).integers(0, 5, 3000)
    shannon = EntropyAnalysis().check().prepare(symbols).run().entropy_[0]
    assert _local('entropy', symbols)[0].mean() == pytest.approx(shannon, abs=1e-9)


def test_local_entropy_gaussian_mean_is_differential_entropy():
    """Test that the mean local entropy of a Gaussian sample is the entropy of the fitted Gaussian."""
    values = np.random.default_rng(3).normal(0.0, 2.0, 10_000)
    expected = np.log2(2 * np.pi * np.e * values.var()) / 2
    assert _local('entropy', values, estimator='gaussian')[0].mean() == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(('history', 'lag'), [(1, 1), (2, 1), (1, 3)])
def test_local_transfer_entropy_mean_is_global(history, lag):
    """Test that the mean local transfer entropy of symbols equals the plug-in transfer entropy."""
    rng = np.random.default_rng(4)
    source = rng.integers(0, 2, 5000)
    target = np.roll(source, 1) ^ (rng.random(5000) < 0.1)
    data = np.column_stack([source, target])
    local = _local('transfer', data, history=history, lag=lag)[0, 1]
    assert local.size == 5000 - max(history, lag)
    expected = _global('transfer', data, history=history, lag=lag)[0, 1]
    assert local.mean() == pytest.approx(expected, abs=1e-9)


def test_local_transfer_entropy_gaussian_detects_coupling():
    """Test that the Gaussian local transfer entropy of a driven autoregression is positive on average."""
    rng = np.random.default_rng(5)
    source = rng.normal(size=5000)
    target = np.zeros(5000)
    for t in range(1, 5000):
        target[t] = 0.5 * target[t - 1] + 0.8 * source[t - 1] + 0.3 * rng.normal()
    local = _local('transfer', np.column_stack([source, target]), estimator='gaussian')
    assert local[0, 1].mean() > 1.0
    assert local[1, 0].mean() == pytest.approx(0.0, abs=0.01)


def test_local_entropy_unknown_estimator():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin, gaussian'):
        LocalInformationAnalysis(estimator='knn').check()


def test_local_entropy_ignores_lag():
    """Test that the local entropy rejects a changed lag, which only the transfer entropy reads."""
    with pytest.raises(AnalysisError, match='does not read lag'):
        LocalInformationAnalysis('entropy', lag=2).check()
