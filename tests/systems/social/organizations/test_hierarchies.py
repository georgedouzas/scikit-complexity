"""Test the `_hierarchies` module of `skcomplex.systems.social.organizations`."""

from skcomplex.systems.social.organizations import (
    bonabeau_hierarchies,
)

from ..conftest import _run, _values


def test_bonabeau_hierarchy_emerges_at_high_density():
    """Test that strengths spread far more at a high density than at a low one."""
    sparse = _values(_run(bonabeau_hierarchies(density=0.05, random_state=0), 100), 'contenders', 'strength')
    dense = _values(_run(bonabeau_hierarchies(density=0.8, random_state=0), 100), 'contenders', 'strength')
    assert dense.std() > 3 * sparse.std()
