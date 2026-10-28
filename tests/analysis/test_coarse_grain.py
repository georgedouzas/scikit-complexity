"""Test the coarse-graining of agents into fields."""

import numpy as np
import pytest

from skcomplex.analysis import CoarseGrainAnalysis, SpectrumAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models import run


def _coarse_grain(positions: object, shape: list[int], **settings: object) -> np.ndarray:
    """Return the field of agents coarse-grained on a grid."""
    return CoarseGrainAnalysis(shape, **settings).check().prepare(positions).run().field_


def test_coarse_grain_density():
    """Test that 10,000 agents coarse-grained on a 32 by 32 grid sum to 10,000 and have a structure factor."""
    rng = np.random.default_rng(6)
    density = _coarse_grain(rng.random((10_000, 2)) * 5, [32, 32], box=[5.0, 5.0])
    assert density.sum() == 10_000
    assert SpectrumAnalysis('structure-factor').check().prepare(density).run().power_.size > 0
    polarization = _coarse_grain(rng.random((100, 2)), [4, 4], box=[1.0, 1.0], values=np.ones((100, 2)))
    assert polarization.shape == (4, 4, 2)
    with pytest.raises(AnalysisError, match='one entry per dimension'):
        CoarseGrainAnalysis([4], box=[1.0, 1.0]).check()


def test_coarse_grain_agents_of_a_trajectory():
    """Test that the walkers of a trajectory give one grid per frame in the box of their space."""
    trajectory = run('run_and_tumble', n_agents=50, settings={'until': 3}).trajectory_
    field = _coarse_grain(trajectory, [5, 5])
    assert field.shape == (4, 5, 5)
    assert np.all(field.sum(axis=(1, 2)) == 50)
    with pytest.raises(AnalysisError, match='needs the sides of the box'):
        CoarseGrainAnalysis([5, 5]).check().prepare(np.zeros((3, 2)))
