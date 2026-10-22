"""Test the agent economies that repeat covered patterns: barter, shops, order books, the commons, and ridesharing."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.models import run
from skcomplex.models.economics.markets import contract_curve, goods_traders


def test_goods_traders_trades_conserve_goods():
    """Test that 50 steps of barter keep the total of every good while utility rises and prices converge."""
    trajectory = run('goods_traders', settings={'until': 50}).trajectory_
    for good in ('sugar', 'spice', 'salt'):
        np.testing.assert_allclose(trajectory[good], trajectory[good][0], rtol=1e-9)
    dispersion = trajectory['exchange.dispersion']
    assert np.nanmean(dispersion[-10:]) < 0.25 * np.nanmean(dispersion[1:6])
    assert trajectory['utility'][-1] > trajectory['utility'][0]


def test_goods_traders_of_two_goods_trade_no_salt():
    """Test that toy traders of two goods hold no salt and still trade."""
    trajectory = run('goods_traders', goods=2, settings={'until': 5}).trajectory_
    assert np.all(trajectory['salt'] == 0)
    assert np.nansum(trajectory['exchange.volume']) > 0


def test_goods_traders_reject_other_numbers_of_goods():
    """Test that the traders barter two or three goods only."""
    with pytest.raises(ValueError, match='2 or 3 goods'):
        goods_traders(goods=4)


@pytest.mark.parametrize(
    ('chocolate', 'cigarettes', 'weights'),
    [((9.0, 1.0), (2.0, 8.0), (0.6, 0.3)), ((5.0, 5.0), (1.0, 9.0), (0.2, 0.7)), ((2.0, 6.0), (7.0, 3.0), (0.5, 0.5))],
)
def test_barter_reaches_the_contract_curve(chocolate, cigarettes, weights):
    """Test that Edgeworth barter ends on the contract curve within 2% of the total endowment."""
    simulation = run('barter', chocolate=chocolate, cigarettes=cigarettes, weights=weights, settings={'until': 5})
    final = simulation.trajectory_.agents('barterers', 5).values
    expected = contract_curve(final['chocolate'][0], *weights, sum(chocolate), sum(cigarettes))
    assert abs(final['cigarettes'][0] - expected) < 0.02 * sum(cigarettes)
    assert final['chocolate'].sum() == pytest.approx(sum(chocolate))
    assert simulation.trajectory_['exchange.volume'][-1] == 0


def test_red_cross_barter_settles_on_one_price():
    """Test that the prices of the barter among many prisoners converge."""
    trajectory = run('barter', n_barterers=100, settings={'until': 20}).trajectory_
    dispersion = trajectory['exchange.dispersion']
    assert np.nanmean(dispersion[-5:]) < 0.1 * dispersion[1]


def test_barter_needs_two_barterers():
    """Test that barter rejects a single barterer."""
    with pytest.raises(ValueError, match='at least two'):
        run('barter', n_barterers=1)


def test_digital_market_closes_shops_and_lowers_prices():
    """Test that shopping online leaves fewer shops, a larger leader, and lower prices than shopping in person."""
    local = run('shops', online=0.0, settings={'until': 100}).trajectory_
    digital = run('shops', online=1.0, settings={'until': 100}).trajectory_
    assert digital['open_shops'][-1] < local['open_shops'][-1]
    assert digital['concentration'][-1] > 2 * local['concentration'][-1]
    assert digital['mean_price'][50:].mean() < local['mean_price'][50:].mean()


def test_order_book_chartists_raise_volatility_and_conserve_holdings():
    """Test that chartists make returns more volatile and that trades keep the total money and shares."""
    returns = {}
    for chartists in (0.0, 0.5):
        trajectory = run('order_book', chartists=chartists, imitation=0.0, settings={'until': 150}).trajectory_
        prices = trajectory['book.price'][1:]
        returns[chartists] = np.diff(np.log(prices))
        traders = trajectory.agents('traders', 150).values
        assert traders['money'].sum() == pytest.approx(400 * 1000.0)
        assert traders['shares'].sum() == pytest.approx(400 * 10.0)
    assert returns[0.5].std() > 1.5 * returns[0.0].std()
    assert abs(stats.trim_mean(returns[0.0], 0.1)) < 0.01


def test_order_book_imitation_drives_out_chartists():
    """Test that traders copy the more successful fundamentalists, so the share of chartists falls."""
    trajectory = run('order_book', chartists=0.5, settings={'until': 150}).trajectory_
    assert trajectory['chartists'][-1] < 0.6 * trajectory['chartists'][0]


def test_commons_collapse():
    """Test that an unregulated meadow with too many herders falls below a tenth of its grass and the herd shrinks."""
    trajectory = run('tragedy_of_the_commons', settings={'until': 60}).trajectory_
    capacity = trajectory['grass'][0]
    collapse = int(np.argmax(trajectory['grass'] < 0.1 * capacity))
    assert trajectory['grass'][collapse] < 0.1 * capacity
    herd = trajectory['herd']
    assert herd[collapse:].min() < 0.5 * herd[: collapse + 1].max()


def test_regulated_commons_keeps_its_grass():
    """Test that a regulator with an adaptive quota keeps the grass above a tenth of its capacity."""
    trajectory = run('tragedy_of_the_commons', quota=2.0, settings={'until': 60}).trajectory_
    assert trajectory['grass'].min() > 0.1 * trajectory['grass'][0]
    assert np.diff(trajectory['quota']).any()


def test_ridesharing_waits_grow_near_saturation():
    """Test that riders wait far longer and every cab is busy once the demand exceeds what the fleet serves."""
    calm = run('ridesharing', demand=0.3, settings={'until': 150}).trajectory_
    rush = run('ridesharing', demand=1.2, settings={'until': 150}).trajectory_
    assert np.nanmean(rush['mean_wait'][100:]) > 5 * np.nanmean(calm['mean_wait'][100:])
    assert rush['utilisation'][100:].mean() > 0.9 > calm['utilisation'][100:].mean()
    assert rush['queue'][-1] > calm['queue'][-1]
