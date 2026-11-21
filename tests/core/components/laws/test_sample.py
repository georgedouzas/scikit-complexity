"""Test the `_sample` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import MoveSpec, ObjectiveSpec, SamplerSpec
from skcomplex.core.components.laws._sample import _SampleLaw
from skcomplex.core.components.spaces import RadiusNeighborhood
from skcomplex.core.errors import ValidationError

SPIN = var('spin')
ENERGY = ObjectiveSpec(-SPIN, RadiusNeighborhood(1, metric='manhattan'))


def test_sample_targets_the_state_of_its_move():
    """Test that a Monte Carlo sample targets the moved state, with or without a sampler."""
    assert _SampleLaw(ENERGY, MoveSpec(SPIN), SamplerSpec('glauber', 2.0))._get_targets() == (SPIN,)
    assert _SampleLaw(ENERGY, MoveSpec(SPIN, 'random-value', values=(0.0, 1.0)))._get_targets() == (SPIN,)


@pytest.mark.parametrize(
    ('move', 'sampler', 'message'),
    [(MoveSpec(SPIN, 'swap'), None, 'Unknown move'), (MoveSpec(SPIN), SamplerSpec('other'), 'Unknown sampler')],
)
def test_sample_checks_its_move_and_sampler(move, sampler, message):
    """Test that a Monte Carlo sample checks its move and its sampler."""
    with pytest.raises(ValidationError, match=message):
        _SampleLaw(ENERGY, move, sampler)._check()
