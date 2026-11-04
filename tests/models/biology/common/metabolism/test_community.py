"""Test the `_community` module of `skcomplex.models.biology.common.metabolism`."""

import pytest

from skcomplex.core.spaces import StaticTime
from skcomplex.core.system import System
from skcomplex.models.biology.common.metabolism import (
    CommunityAnalysis,
    load_network,
)

AEROBIC = 0.8739215


def test_community_steadycom_of_identical_members():
    """Test that identical members share the common rate 0.8739 with abundances that sum to 1."""
    first, second = load_network('textbook', 'first'), load_network('textbook', 'second')
    system = System(elements=[first, second], time=StaticTime())
    analysis = CommunityAnalysis(medium={'EX_glc__D_e': -20.0}).check().prepare(system).run()
    assert analysis.growth_ == pytest.approx(AEROBIC, abs=1e-4)
    assert sum(analysis.abundances_.values()) == pytest.approx(1.0)


def test_community_tradeoff_at_full_growth():
    """Test that the cooperative trade-off at fraction 1 reaches the optimum of the community."""
    first, second = load_network('textbook', 'first'), load_network('textbook', 'second')
    system = System(elements=[first, second], time=StaticTime())
    analysis = CommunityAnalysis('tradeoff', fraction=1.0).check().prepare(system).run()
    assert analysis.growth_ == pytest.approx(AEROBIC, abs=1e-5)
    assert set(analysis.members_) == {'first', 'second'}
