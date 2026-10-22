"""Test that local information measures average to the global measures."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    local_entropy,
    local_mutual_information,
    local_transfer_entropy,
    mutual_information,
    shannon_entropy,
    total_correlation,
    transfer_entropy,
)
from skcomplex.core.base import AnalysisError


def test_local_mutual_information_gaussian_mean_is_global():
    """Test that the mean local mutual information of correlated Gaussians equals the global one within 1e-9."""
    rng = np.random.default_rng(0)
    data = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 1.0]], 50_000)
    local = local_mutual_information(data[:, 0], data[:, 1], estimator='gaussian')
    assert local.shape == (50_000,)
    assert local.mean() == pytest.approx(total_correlation(data, estimator='gaussian'), abs=1e-9)


def test_local_mutual_information_plugin_mean_is_global():
    """Test that the mean local mutual information of symbols equals their plug-in mutual information."""
    rng = np.random.default_rng(1)
    first = rng.integers(0, 3, 5000)
    second = (first + rng.integers(0, 2, 5000)) % 3
    local = local_mutual_information(first, second)
    assert local.mean() == pytest.approx(mutual_information(first, second), abs=1e-9)


def test_local_entropy_mean_is_shannon_entropy():
    """Test that the mean local entropy of symbols equals their Shannon entropy."""
    symbols = np.random.default_rng(2).integers(0, 5, 3000)
    assert local_entropy(symbols).mean() == pytest.approx(shannon_entropy(symbols), abs=1e-9)


def test_local_entropy_gaussian_mean_is_differential_entropy():
    """Test that the mean local entropy of a Gaussian sample is the entropy of the fitted Gaussian."""
    values = np.random.default_rng(3).normal(0.0, 2.0, 10_000)
    expected = np.log2(2 * np.pi * np.e * values.var()) / 2
    assert local_entropy(values, estimator='gaussian').mean() == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(('history', 'lag'), [(1, 1), (2, 1), (1, 3)])
def test_local_transfer_entropy_mean_is_global(history, lag):
    """Test that the mean local transfer entropy of symbols equals the plug-in transfer entropy."""
    rng = np.random.default_rng(4)
    source = rng.integers(0, 2, 5000)
    target = np.roll(source, 1) ^ (rng.random(5000) < 0.1)
    local = local_transfer_entropy(source, target, history=history, lag=lag)
    assert local.size == 5000 - max(history, lag)
    assert local.mean() == pytest.approx(transfer_entropy(source, target, history=history, lag=lag), abs=1e-9)


def test_local_transfer_entropy_gaussian_detects_coupling():
    """Test that the Gaussian local transfer entropy of a driven autoregression is positive on average."""
    rng = np.random.default_rng(5)
    source = rng.normal(size=5000)
    target = np.zeros(5000)
    for t in range(1, 5000):
        target[t] = 0.5 * target[t - 1] + 0.8 * source[t - 1] + 0.3 * rng.normal()
    driven = local_transfer_entropy(source, target, estimator='gaussian').mean()
    reverse = local_transfer_entropy(target, source, estimator='gaussian').mean()
    assert driven > 1.0
    assert reverse == pytest.approx(0.0, abs=0.01)


def test_local_entropy_unknown_estimator():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin, gaussian'):
        local_entropy([0, 1], estimator='knn')
