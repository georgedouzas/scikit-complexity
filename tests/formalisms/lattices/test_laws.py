"""Test the laws of lattice models and the states they change."""

import pytest

from skcomplex.core.base import symbol
from skcomplex.core.space import Neighborhood
from skcomplex.formalisms.lattices import DriveLaw, Energy, LocalRuleLaw, MoveLaw, RelaxationLaw, SampleLaw, SamplerLaw


def test_relaxation_targets_the_states_of_its_relaxation_laws():
    """Test that a relaxation checks its laws and targets the states they change, in order of first appearance."""
    height, grains = symbol('height'), symbol('grains')
    topple = LocalRuleLaw((height, grains), Neighborhood.von_neumann(center=True), lambda z, *n: z)
    settle = LocalRuleLaw(height, Neighborhood.von_neumann(center=True), lambda z, *n: z)
    drop = DriveLaw(height, lambda values, *, random: values + 1)
    law = RelaxationLaw([drop], [topple, settle], lambda *values: True).check()
    assert law.targets == (height, grains)
    assert law.drive[0].targets == (height,)
    assert law.relax[1].targets == (height,)


def test_sample_targets_the_state_of_its_move():
    """Test that a Monte Carlo sample checks its move and sampler and targets the moved state."""
    spin = symbol('spin')
    law = SampleLaw(Energy(-spin, Neighborhood.von_neumann()), MoveLaw.flip(spin), SamplerLaw('glauber', 2.0)).check()
    assert law.targets == (spin,)
    assert law.move.targets == (spin,)
    with pytest.raises(ValueError, match='Unknown sampler'):
        SampleLaw(Energy(-spin, Neighborhood.von_neumann()), MoveLaw.flip(spin), SamplerLaw('other')).check()


@pytest.mark.parametrize(
    'law',
    [
        DriveLaw((symbol('a'), symbol('b')), lambda values, *, random: values),
        MoveLaw((symbol('a'), symbol('b'))),
    ],
)
def test_laws_of_one_lattice_state_reject_several_targets(law):
    """Test that a drive and a move change one lattice state."""
    with pytest.raises(ValueError, match='changes one lattice state'):
        law.check()
