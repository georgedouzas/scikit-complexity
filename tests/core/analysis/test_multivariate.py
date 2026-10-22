"""Test total correlations, O-information, its gradients, and the partial information decomposition."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    dual_total_correlation,
    o_information,
    o_information_gradients,
    partial_information,
    total_correlation,
)
from skcomplex.core.base import AnalysisError

SAMPLES = 20_000


def _bits(seed: int, columns: int) -> np.ndarray:
    """Return fair independent bits, one column per variable."""
    return np.random.default_rng(seed).integers(0, 2, (SAMPLES, columns))


def test_o_information_xor_is_synergistic():
    """Test that X, Y, and Z = X xor Y with fair independent bits have an O-information of -1 bit."""
    bits = _bits(0, 2)
    data = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
    assert o_information(data) == pytest.approx(-1.0, abs=0.01)
    assert o_information_gradients(data) == pytest.approx([-1.0, -1.0, -1.0], abs=0.01)


def test_o_information_copies_is_redundant():
    """Test that three copies of one fair bit have an O-information of +1 bit."""
    bit = _bits(1, 1)
    data = np.column_stack([bit, bit, bit])
    assert o_information(data) == pytest.approx(1.0, abs=0.01)
    assert total_correlation(data) == pytest.approx(2.0, abs=0.01)
    assert dual_total_correlation(data) == pytest.approx(1.0, abs=0.01)


def test_total_correlation_independent_variables_is_zero():
    """Test that independent bits share no information under any measure."""
    data = _bits(2, 4)
    assert total_correlation(data) == pytest.approx(0.0, abs=0.01)
    assert dual_total_correlation(data) == pytest.approx(0.0, abs=0.01)


def test_total_correlation_gaussian_matches_closed_form():
    """Test that the Gaussian total correlation of two correlated Gaussians is `-log2(1 - r^2) / 2`."""
    rng = np.random.default_rng(3)
    correlation = 0.6
    data = rng.multivariate_normal([0.0, 0.0], [[1.0, correlation], [correlation, 1.0]], SAMPLES)
    sample = np.corrcoef(data, rowvar=False)[0, 1]
    expected = -np.log2(1 - sample**2) / 2
    assert total_correlation(data, estimator='gaussian') == pytest.approx(expected, abs=1e-9)
    assert dual_total_correlation(data, estimator='gaussian') == pytest.approx(expected, abs=1e-9)
    assert o_information(data, estimator='gaussian') == pytest.approx(0.0, abs=1e-9)


def test_o_information_gaussian_common_driver_is_redundant():
    """Test that three noisy copies of one Gaussian are redundant under the Gaussian and neighbour estimators."""
    rng = np.random.default_rng(4)
    driver = rng.normal(size=4000)
    data = np.column_stack([driver + 0.3 * rng.normal(size=driver.size) for _ in range(3)])
    gaussian = o_information(data, estimator='gaussian')
    assert gaussian > 0.5
    assert o_information(data, estimator='knn') == pytest.approx(gaussian, abs=0.15)


def test_o_information_gaussian_sum_is_synergistic():
    """Test that two independent Gaussians and their noisy sum are synergistic under the Gaussian estimator."""
    rng = np.random.default_rng(5)
    first, second = rng.normal(size=(2, SAMPLES))
    data = np.column_stack([first, second, first + second + 0.1 * rng.normal(size=SAMPLES)])
    assert o_information(data, estimator='gaussian') < -0.5


def test_total_correlation_unknown_estimator():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin, gaussian, knn'):
        total_correlation(_bits(6, 2), estimator='binned')


def test_partial_information_and_gate():
    """Test that Z = X and Y with the minimum mutual information has 0.311 bits of redundancy and 0.5 of synergy."""
    bits = _bits(7, 2)
    parts = partial_information(bits[:, 0], bits[:, 1], bits[:, 0] & bits[:, 1])
    assert parts.redundancy == pytest.approx(0.311, abs=0.01)
    assert parts.synergy == pytest.approx(0.5, abs=0.01)
    assert parts.unique_first == pytest.approx(0.0, abs=0.01)
    assert parts.unique_second == pytest.approx(0.0, abs=0.01)
    assert parts.total == pytest.approx(0.811, abs=0.01)


def test_partial_information_redundancy_measures():
    """Test that copying one source gives unique information under Williams and Beer's measure."""
    bits = _bits(8, 2)
    parts = partial_information(bits[:, 0], bits[:, 1], bits[:, 0], redundancy='williams_beer')
    assert parts.unique_first == pytest.approx(1.0, abs=0.01)
    assert parts.synergy == pytest.approx(0.0, abs=0.01)


def test_partial_information_unknown_redundancy():
    """Test that an unknown redundancy measure raises an error naming the known ones."""
    bits = _bits(9, 2)
    with pytest.raises(AnalysisError, match='mmi'):
        partial_information(bits[:, 0], bits[:, 1], bits[:, 0], redundancy='minimum')
