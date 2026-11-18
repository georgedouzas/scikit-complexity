"""Test the `_flow_term` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws._flow_term import FlowTermLaw
from skcomplex.core.components.laws.utils import shift_member
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

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


def test_get_terms_of_an_increment_flow_are_increments():
    """Test that an increment update that maps stocks to signs gives one increment term per stock."""
    full, empty = var('full empty')
    flow = UpdateLaw({full: -1, empty: +1}, 0.5 * full, increment=True, label='pour')
    terms = FlowTermLaw._get_terms(flow, 'pouring')
    assert [(term.target, term.value, term.sign, term.increment) for term in terms] == [
        (full, -0.5 * full, -1, True),
        (empty, 0.5 * full, 1, True),
    ]


def test_get_terms_refuses_an_update_flow_that_is_no_increment():
    """Test that an update law that maps stocks but sets the next value asks for an increment."""
    full = var('full')
    with pytest.raises(ValidationError, match='increment=True'):
        FlowTermLaw._get_terms(UpdateLaw({full: -1}, 0.5 * full, label='drain'), 'draining')


def test_increment_flow_steps_in_discrete_time_and_records_its_amount():
    """Test that a tank drained by half its level every step halves and records the amount it lost."""
    drain = Interaction(
        'drain',
        roles={'target': 'tank'},
        laws=lambda s: [UpdateLaw({s.target.level: -1}, s.target.level / 2, increment=True, label='drain')],
    )
    system = System(elements=[Element('tank', values={'level': 8.0})], interactions=[drain]).prepare()
    trajectory = system.simulate(until=3)
    assert system.form_ == 'discrete map'
    np.testing.assert_allclose(trajectory.get_values('tank.level'), [8.0, 4.0, 2.0, 1.0])
    np.testing.assert_allclose(trajectory.get_values('tank.drain'), [4.0, 2.0, 1.0, 0.5])


@pytest.mark.parametrize(('boundary', 'expected'), [('periodic', [3.0, 2.0, 1.0]), ('fixed', [0.0, 2.0, 4.0])])
def test_increment_flow_to_the_next_member_moves_every_amount_along_the_group(boundary, expected):
    """Test that `{s.n: -1, shift_member(s.n, 1): +1}` moves each site's amount to the next, none past a fixed end."""

    def laws(s):
        n = s.target.n
        return [UpdateLaw({n: -1, shift_member(n, 1, boundary=boundary): +1}, n, increment=True, label='hops')]

    hops = Interaction('hops', roles={'target': 'sites'}, laws=laws)
    sites = Element('sites', values={'n': np.array([2.0, 1.0, 3.0])}, space=LatticeSpace(3, boundary=boundary))
    trajectory = System(elements=[sites], interactions=[hops]).prepare().simulate(until=1)
    np.testing.assert_allclose(trajectory.get_values('sites.n')[-1], expected)
    assert np.sum(trajectory.get_values('sites.hops')[0]) == pytest.approx(6.0 if boundary == 'periodic' else 3.0)
