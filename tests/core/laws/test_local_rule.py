"""Test the `_local_rule` module of `skcomplex.core.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import LocalRuleLaw
from skcomplex.core.spaces import BoxNeighborhood, CayleySpace, ContinuousTime, DiscreteTime, LatticeSpace
from skcomplex.errors import ValidationError
from skcomplex.specs import TotalisticSpec


def test_local_rule_targets_one_state_or_several_together():
    """Test that a local rule targets one lattice state, or a tuple of them that it updates together."""
    height, grains = var('height grains')
    assert LocalRuleLaw(height, BoxNeighborhood(), TotalisticSpec({2: 1})).check().prepare().targets_ == (height,)
    together = LocalRuleLaw((height, grains), BoxNeighborhood(), lambda z, *n: (z, z)).check().prepare()
    assert together.targets_ == (height, grains)


def test_local_rule_requires_discrete_time_on_sites():
    """Test that a local rule runs in discrete time on a lattice, a network, or a Cayley graph."""
    requirements = LocalRuleLaw(var('cell'), BoxNeighborhood(), TotalisticSpec({2: 1}))._get_requirements()
    assert requirements.time == (DiscreteTime,)
    assert LatticeSpace in requirements.space
    assert CayleySpace in requirements.space
    assert ContinuousTime not in requirements.time


def test_local_rule_rejects_an_unknown_schedule():
    """Test that a local rule rejects a schedule it does not know."""
    with pytest.raises(ValidationError, match='Unknown schedule'):
        LocalRuleLaw(var('cell'), BoxNeighborhood(), TotalisticSpec({2: 1}), schedule='parallel').check()
