"""Test symbolisation and information measures against known values."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    block_entropy,
    correlation_information,
    entropy_rate,
    excess_entropy,
    kl_divergence,
    lempel_ziv_complexity,
    mutual_information,
    permutation_entropy,
    shannon_entropy,
    symbolize,
    transfer_entropy,
)
from skcomplex.core.base import AnalysisError


def test_periodic_sequence():
    """Test that the repeated sequence 0011 has entropy rate 0 and excess entropy 2 bits."""
    sequence = [0, 0, 1, 1] * 500
    assert entropy_rate(sequence, 6) == pytest.approx(0.0, abs=1e-6)
    assert excess_entropy(sequence, 6) == pytest.approx(2.0, abs=1e-6)
    assert correlation_information(sequence, 4) == pytest.approx([0.0, 1.0, 0.0], abs=1e-5)


def test_random_sequence_has_full_entropy_rate():
    """Test that fair random bits have an entropy rate near 1 bit and no excess entropy."""
    bits = np.random.default_rng(0).integers(0, 2, 100_000)
    assert entropy_rate(bits, 6) == pytest.approx(1.0, abs=0.01)
    assert excess_entropy(bits, 6) == pytest.approx(0.0, abs=0.05)


def test_information_flows_from_driver():
    """Test that a series copying a random series with lag 1 receives 1 bit from it and gives none back."""
    driver = np.random.default_rng(1).integers(0, 2, 20_000)
    follower = np.roll(driver, 1)
    assert transfer_entropy(driver, follower) == pytest.approx(1.0, abs=0.01)
    assert transfer_entropy(follower, driver) == pytest.approx(0.0, abs=0.01)


def test_entropies_and_divergences():
    """Test Shannon entropy, KL divergence, mutual information, and two-dimensional block entropy."""
    assert shannon_entropy([0.5, 0.25, 0.25], probabilities=True) == pytest.approx(1.5)
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0)
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float('inf')
    independent = np.random.default_rng(2).integers(0, 2, (2, 50_000))
    assert mutual_information(*independent) == pytest.approx(0.0, abs=0.001)
    checkerboard = np.indices((40, 40)).sum(axis=0) % 2
    assert block_entropy(checkerboard, 2) == pytest.approx(1.0)


def test_symbolisation_and_permutations():
    """Test quantile, bin, and ordinal symbols and the permutation entropy of monotone and random series."""
    assert symbolize([1.0, 2.0, 3.0, 4.0], 'bins', 2).tolist() == [0, 0, 1, 1]
    assert np.bincount(symbolize(np.random.default_rng(3).normal(size=1000), 'quantiles', 4)).tolist() == [250] * 4
    assert permutation_entropy(np.arange(50.0)) == 0.0
    assert permutation_entropy(np.random.default_rng(4).normal(size=50_000)) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(AnalysisError, match='Unknown symbolisation'):
        symbolize([1.0], 'words')


def test_lempel_ziv_complexity():
    """Test the LZ76 phrase count of Kaspar and Schuster's example and normalised complexity of random bits."""
    assert lempel_ziv_complexity([int(bit) for bit in '0001101001000101'], normalize=False) == 6
    random_bits = np.random.default_rng(5).integers(0, 2, 5000)
    assert lempel_ziv_complexity(random_bits) == pytest.approx(1.0, abs=0.1)
