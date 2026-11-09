"""Test the `_cascade` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import ArrayUpdateLaw, CascadeLaw, LocalRuleLaw
from skcomplex.core.components.spaces import BallNeighborhood
from skcomplex.core.errors import ValidationError


def test_cascade_targets_the_states_of_the_laws_that_relax():
    """Test that a cascade targets the states the laws that relax the lattice change, each once in order."""
    height, grains = var('height grains')
    topple = LocalRuleLaw((height, grains), BallNeighborhood(metric='manhattan', center='first'), lambda z, *n: (z, z))
    settle = LocalRuleLaw(height, BallNeighborhood(metric='manhattan', center='first'), lambda z, *n: z)
    drop = ArrayUpdateLaw(height, lambda values, *, random: values + 1)
    law = CascadeLaw([drop], [topple, settle], lambda *values: True).prepare()
    assert law.targets_ == (height, grains)


def test_cascade_checks_every_law_it_holds():
    """Test that checking a cascade checks every law it holds."""
    height = var('height')
    settle = LocalRuleLaw(
        height,
        BallNeighborhood(metric='manhattan', center='first'),
        lambda z, *n: z,
        schedule='parallel',
    )
    with pytest.raises(ValidationError, match='Unknown schedule'):
        CascadeLaw([], [settle], lambda *values: True).prepare()
    with pytest.raises(ValidationError, match='callable'):
        CascadeLaw([ArrayUpdateLaw(height, None)], [], lambda *values: True).prepare()
