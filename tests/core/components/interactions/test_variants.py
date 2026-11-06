"""Test the variants of the general interactions that take names and functions, or names alone."""

import pytest
from sage.all__sagemath_symbolics import SR

from skcomplex.core.components.interactions import LifeCycleInteraction, MovementInteraction, SamplingInteraction
from skcomplex.core.components.laws import EmissionLaw
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PartnerSpec


@pytest.mark.parametrize(
    ('interaction', 'message'),
    [
        (MovementInteraction('a', target='b', step=PartnerSpec()), 'Unknown step .*a function'),
        (LifeCycleInteraction('a', target='b', placement=PartnerSpec()), 'Unknown placement .*a function'),
        (
            SamplingInteraction('a', target='b', state='x', objective=lambda s: 0.0, move=PartnerSpec()),
            'Unknown move .*a function',
        ),
        (LifeCycleInteraction('a', target='b', pick=PartnerSpec()), "Unknown pick .*'first', 'random'"),
    ],
)
def test_variant_refuses_a_spec_and_lists_what_it_takes(interaction, message):
    """Test that a step, a placement, a move, and a pick refuse a spec, and list their names and functions."""
    with pytest.raises(ValidationError, match=message):
        interaction.prepare()


def test_emission_law_says_its_matrix_is_symbolic_only():
    """Test that an emission refuses a function, and names the laws that take one."""
    x, y = SR.var('x'), SR.var('y')
    with pytest.raises(ValidationError, match=r'symbolic only.*TransitionLaw'):
        EmissionLaw(x, y, lambda t, values: [[1.0]]).prepare()
