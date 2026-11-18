"""Test the `_flow_term` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws._flow_term import FlowTermLaw
from skcomplex.core.errors import ValidationError

alive, dead, d = var('alive dead d')


def test_a_flow_splits_into_one_term_per_stock_with_its_sign():
    """Test that a flow gives `-f` with the sign -1 on the stock it leaves and `f` with +1 on the one it enters."""
    terms = FlowTermLaw._get_terms(RateLaw({alive: -1, dead: +1}, d * alive, label='deaths'), 'mortality')
    assert [(term.target, term.value, term.sign, term.label) for term in terms] == [
        (alive, -d * alive, -1, 'deaths'),
        (dead, d * alive, 1, 'deaths'),
    ]


def test_a_rate_of_one_state_is_kept():
    """Test that a rate law whose target is no mapping is returned as it is."""
    law = RateLaw(alive, -d * alive)
    assert FlowTermLaw._get_terms(law, 'mortality') == [law]


@pytest.mark.parametrize(
    ('law', 'match'),
    [
        (RateLaw({alive: -1, dead: +1}, d * alive), 'no label'),
        (RateLaw({alive: -1, dead: -1}, d * alive, label='deaths'), 'signs'),
        (RateLaw({alive: 2}, d * alive, label='deaths'), 'signs'),
        (RateLaw({}, d * alive, label='deaths'), 'signs'),
    ],
)
def test_a_flow_without_label_or_with_wrong_signs_is_rejected(law, match):
    """Test that a flow needs a label and leaves at most one stock and enters at most one."""
    with pytest.raises(ValidationError, match=match):
        FlowTermLaw._get_terms(law, 'mortality')
