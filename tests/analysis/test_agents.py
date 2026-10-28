"""Test the coarse-graining of agents into fields."""

import numpy as np
import pytest

from skcomplex.analysis import SpectrumAnalysis, coarse_grain
from skcomplex.core.errors import AnalysisError


def test_coarse_grain_density():
    """Test that 10,000 agents coarse-grained on a 32 by 32 grid sum to 10,000 and have a structure factor."""
    rng = np.random.default_rng(6)
    density = coarse_grain(rng.random((10_000, 2)) * 5, [5.0, 5.0], [32, 32])
    assert density.sum() == 10_000
    assert SpectrumAnalysis('structure-factor').check().prepare(density).run().power_.size > 0
    polarization = coarse_grain(rng.random((100, 2)), [1.0, 1.0], [4, 4], values=np.ones((100, 2)))
    assert polarization.shape == (4, 4, 2)
    with pytest.raises(AnalysisError, match='one entry per dimension'):
        coarse_grain(rng.random((5, 2)), [1.0, 1.0], [4])
