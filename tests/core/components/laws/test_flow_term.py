"""Test the `_flow_term` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.analysis import loops, symbolic
from skcomplex.core.components.elements import Element, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws._flow_term import FlowTermLaw
from skcomplex.core.components.laws.utils import shift_member, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.runtime import ConservationObserver
from skcomplex.core.errors import ConservationError, ValidationError
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


class Switch(Element):
    """Switch whose state `lit` is off or on."""

    def __init__(self, label: str, *, lit: Annotated[int, {0, 1}] = 0) -> None:
        super().__init__(label)
        self.lit = lit


def test_increment_flow_moves_a_value_of_a_finite_set():
    """Test that a flow in discrete time moves a value allowed only 0 and 1, since it adds no rate."""
    flip = Interaction(
        'flip',
        roles={'off': 'off', 'on': 'on'},
        laws=lambda s: [UpdateLaw({s.off.lit: -1, s.on.lit: 1}, s.off.lit, increment=True, label='flip')],
    )
    trajectory = System(elements=[Switch('off', lit=1), Switch('on')], interactions=[flip]).prepare().simulate(until=1)
    assert [trajectory.get_values('off.lit')[-1], trajectory.get_values('on.lit')[-1]] == [0, 1]


class LedgerElement(Element):
    """Two accounts of money and a counter of steps."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[float, Quantity('money')] = 10.0,
        b: Annotated[float, Quantity('money')] = 0.0,
        c: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.c = c


def _book(s):
    """Return a payment of a tenth of `a` to `b`, a loan of 1 into `a`, and a counter, once per step."""
    e = s.target
    return [
        UpdateLaw({e.a: -1, e.b: +1}, 0.1 * e.a, increment=True, label='pay'),
        UpdateLaw({e.a: +1}, 1.0, increment=True, label='lend'),
        UpdateLaw(e.c, e.c + 1),
    ]


def test_discrete_flows_are_per_step_and_join_the_updates():
    """Test that in a discrete map a flow moves its amount once per step and money changes only by lending."""
    book = Interaction('book', roles={'target': 'ledger'}, laws=_book)
    trajectory = System(elements=[LedgerElement('ledger')], interactions=[book]).prepare().simulate(until=5, dt=1)
    assert trajectory.get_values('ledger.a').tolist() == pytest.approx([10.0] * 6)
    assert trajectory.get_values('ledger.b').tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert trajectory.get_values('__boundary.money')[-1] == pytest.approx(5.0)
    assert trajectory.get_values('ledger.c')[-1] == pytest.approx(5.0)


class PeopleElement(Element):
    """Population of people with a birth rate `b` and a death rate `d`."""

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[float, Quantity('people')] = 10.0,
        b: float = 0.03,
        d: float = 0.01,
    ) -> None:
        super().__init__(label)
        self.n = n
        self.b = b
        self.d = d


def _life(s):
    """Return births and deaths in proportion to the population."""
    e = s.target
    return [RateLaw({e.n: +1}, e.b * e.n, label='births'), RateLaw({e.n: -1}, e.d * e.n, label='deaths')]


def test_population_births_dominate_and_set_the_net_inflow():
    """Test that births dominate deaths, and that the net inflow of people is births minus deaths."""
    system = System(
        elements=[PeopleElement('pop')],
        interactions=[Interaction('life', roles={'target': 'pop'}, laws=_life)],
    )
    dominance = loops(system.prepare().simulate(until=5, dt=1), kind='dominance')
    assert dominance.gains[-1] == pytest.approx([0.03, -0.01])
    assert dominance.polarities[int(dominance.dominant[-1])] == 'reinforcing'
    assert str(symbolic(system, kind='conservation').inflows['people']) == 'b_pop*n_pop - d_pop*n_pop'


class TubElement(Element):
    """Tub of water, a quantity that only flows."""

    def __init__(self, label: str, *, water: Annotated[float, Quantity('water')] = 0.0) -> None:
        super().__init__(label)
        self.water = water


def _clipped(s):
    """Return an inflow and an update that clips the level at 2, which destroys water."""
    water = s.target.water
    return [RateLaw({water: +1}, 1.0, label='faucet'), UpdateLaw(water, where(water > 2, 2, water))]


def test_conservation_observer_detects_drift_beyond_the_flows():
    """Test that clipping a stock outside its flows destroys water, which the conservation observer reports."""
    clip = Interaction('clip', roles={'target': 'tub'}, laws=_clipped)
    system = System(elements=[TubElement('tub')], interactions=[clip]).prepare()
    with pytest.raises(ConservationError, match="'water' drifted"):
        system.simulate(until=5, dt=1, observers=[ConservationObserver()])
