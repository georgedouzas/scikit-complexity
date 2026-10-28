"""Test symbolisation and information measures against known values."""

import numpy as np
import pytest

from skcomplex.analysis import (
    correlation_information,
    kl_divergence,
    mutual_information,
    symbolize,
    transfer_entropy,
)
from skcomplex.core.errors import AnalysisError


def test_periodic_sequence():
    """Test that the repeated sequence 0011 has its correlation information at length 3."""
    sequence = [0, 0, 1, 1] * 500
    assert correlation_information(sequence, 4) == pytest.approx([0.0, 1.0, 0.0], abs=1e-5)


def test_information_flows_from_driver():
    """Test that a series copying a random series with lag 1 receives 1 bit from it and gives none back."""
    driver = np.random.default_rng(1).integers(0, 2, 20_000)
    follower = np.roll(driver, 1)
    assert transfer_entropy(driver, follower) == pytest.approx(1.0, abs=0.01)
    assert transfer_entropy(follower, driver) == pytest.approx(0.0, abs=0.01)


def test_entropies_and_divergences():
    """Test the KL divergence and the mutual information of independent symbols."""
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0)
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float('inf')
    independent = np.random.default_rng(2).integers(0, 2, (2, 50_000))
    assert mutual_information(*independent) == pytest.approx(0.0, abs=0.001)


def test_symbolisation_and_permutations():
    """Test quantile and bin symbols and the error of an unknown symbolisation."""
    assert symbolize([1.0, 2.0, 3.0, 4.0], 'bins', 2).tolist() == [0, 0, 1, 1]
    assert np.bincount(symbolize(np.random.default_rng(3).normal(size=1000), 'quantiles', 4)).tolist() == [250] * 4
    with pytest.raises(AnalysisError, match='Unknown symbolisation'):
        symbolize([1.0], 'words')
