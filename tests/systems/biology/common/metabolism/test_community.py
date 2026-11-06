"""Test the `_community` module of `skcomplex.systems.biology.common.metabolism`."""

import pytest

from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.systems.biology.common.metabolism import (
    CommunityAnalysis,
    ReactionSpec,
    load_network,
    metabolic_network_type,
)

AEROBIC = 0.8739215


def test_community_steadycom_of_identical_members():
    """Test that identical members share the common rate 0.8739 with abundances that sum to 1."""
    first, second = load_network('textbook', 'first'), load_network('textbook', 'second')
    system = System(elements=[first, second], time=StaticTime())
    analysis = CommunityAnalysis(medium={'EX_glc__D_e': -20.0}).prepare(system).run()
    assert analysis.growth_ == pytest.approx(AEROBIC, abs=1e-4)
    assert sum(analysis.abundances_.values()) == pytest.approx(1.0)


@pytest.mark.parametrize('rate', [7.0, 50.0])
def test_community_steadycom_finds_the_rate_at_any_scale(rate):
    """Test a rate that failed at the boundary of the bisection, and one that a limit of 10 would clip."""
    reactions = {'EX_a': ReactionSpec({'a': 1.0}, upper=rate), 'grow': ReactionSpec({'a': -1.0})}
    element_type = metabolic_network_type('ScaledMemberElement', reactions, {'grow': 1.0})
    system = System(elements=[element_type('one'), element_type('two')], time=StaticTime())
    analysis = CommunityAnalysis().prepare(system).run()
    assert analysis.growth_ == pytest.approx(rate, rel=1e-6)
    assert sum(analysis.abundances_.values()) == pytest.approx(1.0)


def test_community_tradeoff_at_full_growth():
    """Test that the cooperative trade-off at fraction 1 reaches the optimum of the community."""
    first, second = load_network('textbook', 'first'), load_network('textbook', 'second')
    system = System(elements=[first, second], time=StaticTime())
    analysis = CommunityAnalysis('tradeoff', fraction=1.0).prepare(system).run()
    assert analysis.growth_ == pytest.approx(AEROBIC, abs=1e-5)
    assert set(analysis.members_) == {'first', 'second'}


def test_community_steadycom_keeps_its_reference_values():
    """Test the common rate against the value recorded before the element moved to declared states."""
    reactions = {'EX_a': ReactionSpec({'a': 1.0}, upper=7.0), 'grow': ReactionSpec({'a': -1.0})}
    element_type = metabolic_network_type('ReferenceMemberElement', reactions, {'grow': 1.0})
    system = System(elements=[element_type('one'), element_type('two')], time=StaticTime())
    analysis = CommunityAnalysis().prepare(system).run()
    assert analysis.growth_ == pytest.approx(6.99999975040555, rel=1e-9)
