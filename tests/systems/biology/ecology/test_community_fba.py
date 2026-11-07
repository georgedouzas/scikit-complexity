"""Test the `_community_fba` module of `skcomplex.systems.biology.ecology`."""

import pytest

from skcomplex.systems import create_simulation
from skcomplex.systems.biology.common.metabolism import community_growth
from skcomplex.systems.biology.ecology import community_fba

GROWTH = 'v_Biomass_Ecoli_core'


def _growth(**arguments):
    """Return the community growth and the growth of the members of a run."""
    trajectory = create_simulation('community_fba', **arguments).prepare().run().trajectory_
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
    """Test that two copies of the system share a medium, each with a glucose uptake of 10."""
    exchanges = {'EX_glc__D_e': -10.0}
    community, members = _growth(knockouts=('', ''), exchanges=exchanges)
    assert members == pytest.approx([0.8739, 0.8739], abs=1e-4)
    assert community == pytest.approx(0.8739, abs=1e-4)
    result = community_growth(community_fba(knockouts=('', ''), exchanges=exchanges), medium={'EX_glc__D_e': -10.0})
    assert result.growth == pytest.approx(0.8739, abs=1e-4)
    assert sum(result.abundances.values()) == pytest.approx(1.0)


def test_community_fba_grows_the_members_at_one_rate():
    """Test that balanced growth gives both members one rate, no faster than joint FBA."""
    joint, _ = _growth()
    community, members = _growth(common_rate=True)
    assert members[0] == pytest.approx(members[1])
    assert community == pytest.approx(members[0])
    assert 0 < community <= joint + 1e-9


def test_community_fba_keeps_its_reference_values():
    """Test the medium of a community against the value recorded before it moved to declared states."""
    community, _ = _growth(knockouts=('', ''), exchanges={'EX_glc__D_e': -10.0})
    assert community == pytest.approx(0.8739215069684299, rel=1e-9)
