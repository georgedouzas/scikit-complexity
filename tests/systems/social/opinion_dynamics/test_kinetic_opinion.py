"""Test the `_kinetic_opinion` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem


def test_opinions_reach_consensus_or_polarise():
    """Test that a wide confidence leads to consensus and a narrow one with repulsion to polarisation."""
    consensus = simulate(KineticOpinionSystem(confidence=6, repulsion=0.0), until=60, dt=60.0, random_state=0)
    polarised = simulate(KineticOpinionSystem(confidence=1, repulsion=0.2), until=60, dt=60.0, random_state=0)
    assert consensus.get_values('opinion_variance')[-1] < 0.1 < consensus.get_values('opinion_variance')[0]
    assert polarised.get_values('opinion_variance')[-1] > 0.9
    extremes = np.asarray(polarised.get_values('opinions.x')[-1])[[0, -1]].sum()
    assert extremes > 0.9
