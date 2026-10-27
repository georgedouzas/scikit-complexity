"""Test the `_relaxation` module of `skcomplex.core.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.errors import ValidationError
from skcomplex.core.laws import DriveLaw, LocalRuleLaw, RelaxationLaw
from skcomplex.core.spaces import VonNeumannNeighborhood


def test_relaxation_targets_the_states_of_its_relaxation_laws():
    """Test that a relaxation targets the states its relaxation laws change, each once in order of appearance."""
    height, grains = var('height grains')
    topple = LocalRuleLaw((height, grains), VonNeumannNeighborhood(center=True), lambda z, *n: (z, z))
    settle = LocalRuleLaw(height, VonNeumannNeighborhood(center=True), lambda z, *n: z)
    drop = DriveLaw(height, lambda values, *, random: values + 1)
    law = RelaxationLaw([drop], [topple, settle], lambda *values: True).check().prepare()
    assert law.targets_ == (height, grains)


def test_relaxation_checks_its_drive_and_relaxation_laws():
    """Test that checking a relaxation checks every law it holds."""
    height = var('height')
    settle = LocalRuleLaw(height, VonNeumannNeighborhood(center=True), lambda z, *n: z, schedule='parallel')
    with pytest.raises(ValidationError, match='Unknown schedule'):
        RelaxationLaw([], [settle], lambda *values: True).check()
    with pytest.raises(ValidationError, match='callable'):
        RelaxationLaw([DriveLaw(height, None)], [], lambda *values: True).check()
