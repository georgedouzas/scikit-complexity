"""Test the partial information decomposition of two sources about a target."""

import numpy as np
import pytest

from skcomplex.core.analysis import partial_information
from skcomplex.core.errors import AnalysisError

SAMPLES = 20_000


def _bits(seed: int, columns: int) -> np.ndarray:
    """Return fair independent bits, one column per variable."""
    return np.random.default_rng(seed).integers(0, 2, (SAMPLES, columns))


def _parts(first: np.ndarray, second: np.ndarray, target: np.ndarray, **settings: object) -> partial_information:
    """Return the run decomposition of what two sources tell about a target."""
    source = np.column_stack([first, second, target])
    return partial_information(source, **settings)


def test_partial_information_and_gate():
    """Test that Z = X and Y with the minimum mutual information has 0.311 bits of redundancy and 0.5 of synergy."""
    bits = _bits(7, 2)
    parts = _parts(bits[:, 0], bits[:, 1], bits[:, 0] & bits[:, 1])
    assert parts.redundancy == pytest.approx(0.311, abs=0.01)
    assert parts.synergy == pytest.approx(0.5, abs=0.01)
    assert parts.unique_first == pytest.approx(0.0, abs=0.01)
    assert parts.unique_second == pytest.approx(0.0, abs=0.01)
    assert parts.total == pytest.approx(0.811, abs=0.01)


def test_partial_information_redundancy_measures():
    """Test that copying one source gives unique information under Williams and Beer's measure."""
    bits = _bits(8, 2)
    parts = _parts(bits[:, 0], bits[:, 1], bits[:, 0], redundancy='williams_beer')
    assert parts.unique_first == pytest.approx(1.0, abs=0.01)
    assert parts.synergy == pytest.approx(0.0, abs=0.01)


def test_partial_information_unknown_redundancy():
    """Test that an unknown redundancy measure raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='mmi'):
        partial_information([0, 1, 0, 1], redundancy='minimum')


def test_partial_information_needs_three_series():
    """Test that a source without two sources and a target raises an error."""
    with pytest.raises(AnalysisError, match='two sources and a target'):
        partial_information(_bits(9, 2))
