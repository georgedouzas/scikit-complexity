"""Test the `_kinetic_opinion` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np

from ...conftest import _trajectory


def test_opinions_reach_consensus_or_polarise():
    """Test that a wide confidence leads to consensus and a narrow one with repulsion to polarisation."""
    consensus = _trajectory('kinetic_opinion', 60, 60.0, confidence=6, repulsion=0.0)
    polarised = _trajectory('kinetic_opinion', 60, 60.0, confidence=1, repulsion=0.2)
    assert consensus.values('opinion_variance')[-1] < 0.1 < consensus.values('opinion_variance')[0]
    assert polarised.values('opinion_variance')[-1] > 0.9
    extremes = np.asarray(polarised.values('opinions.f')[-1])[[0, -1]].sum()
    assert extremes > 0.9
