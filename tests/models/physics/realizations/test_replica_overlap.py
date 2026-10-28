"""Test the `_replica_overlap` module of `skcomplex.models.physics.realizations`."""

import numpy as np

from skcomplex.analysis import CorrelationAnalysis
from skcomplex.models.physics.realizations import ReplicaOverlapAnalysis


def test_replica_overlap_equals_the_overlap():
    """Test that the replica overlap equals the correlation analysis of kind overlap of every frame."""
    replicas = np.random.default_rng(0).choice([-1, 1], (2, 5, 8, 8))
    overlap = ReplicaOverlapAnalysis().check().prepare(replicas).run().overlap_
    general = CorrelationAnalysis('overlap').check().prepare(replicas).run().overlap_
    np.testing.assert_array_equal(overlap, general)
    assert overlap.shape == (5,)
