"""Test the `_community_fba` module of `skcomplex.models.biology.ecology`."""

import pytest

from skcomplex.models import run
from skcomplex.models.biology.common.metabolism import CommunityAnalysis
from skcomplex.models.biology.ecology import community_fba

GROWTH = 'v_Biomass_Ecoli_core'


def _growth(**arguments):
    """Return the community growth and the growth of the members of a run."""
    trajectory = run('community_fba', **arguments).trajectory_
    members = [
        float(trajectory.values(f'member_{position}.{GROWTH}')[0])
        for position in range(len(arguments.get('knockouts', (0, 0))))
    ]
    return float(trajectory.values('medium.objective_0')[0]), members


def test_cross_feeding_mutants_grow_only_together():
    """Test that each mutant alone has no growth, and that together both grow."""
    community, members = _growth()
    assert community > 0.5
    assert all(item > 0.1 for item in members)
    for knockout in ('ICDHyr', 'NH4t'):
        alone, _ = _growth(knockouts=(knockout,), abundances=(1.0,))
        assert alone == pytest.approx(0.0, abs=1e-6)


def test_identical_members_grow_at_the_rate_of_one():
    """Test that two copies of the model share a medium, each with a glucose uptake of 10."""
    exchanges = {'EX_glc__D_e': -10.0}
    community, members = _growth(knockouts=('', ''), exchanges=exchanges)
    assert members == pytest.approx([0.8739, 0.8739], abs=1e-4)
    assert community == pytest.approx(0.8739, abs=1e-4)
    analysis = CommunityAnalysis(medium={'EX_glc__D_e': -10.0})
    analysis = analysis.check().prepare(community_fba(knockouts=('', ''), exchanges=exchanges)).run()
    assert analysis.growth_ == pytest.approx(0.8739, abs=1e-4)
    assert sum(analysis.abundances_.values()) == pytest.approx(1.0)
