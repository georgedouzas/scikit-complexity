"""Test the economics models of the library: macroeconomics, markets, finance, resources, and innovation."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.models import describe
from skcomplex.models.economics.econophysics import boltzmann_wealth
from skcomplex.models.economics.finance import (
    arbitrage_traders,
    bouchaud_mezard_wealth,
    correlated_geometric_brownian_motion,
    debtrank,
    gibrat_process,
    kesten_process,
    multiplicative_process,
)
from skcomplex.models.economics.innovation import (
    arthur_polak,
    creative_phase_transition,
    nk_technology,
    phone_adoption,
)
from skcomplex.models.economics.macroeconomics import capital_stock, exchange_rates, toy_firm_investment
from skcomplex.models.economics.markets import (
    car_dealer,
    car_wash,
    cournot_duopoly,
    produce_market,
    riccati_prices,
)
from skcomplex.models.economics.resources import fishery_capital, oil_economy


def _run(system, until, **settings):
    """Return the trajectory of a system run with a fixed random state."""
    return Simulation(system, random_state=0, **settings).run(until=until).trajectory_


def _agents(trajectory, group, frame):
    """Return the states of a group's agents at a frame."""
    return trajectory.agents(group, frame).values


def _truth_table(arity, function):
    """Return the truth table of a Boolean function, bit `r` holding its value on the input bits of row `r`."""
    return sum(int(function(*(bool(row >> pin & 1) for pin in range(arity)))) << row for row in range(2**arity))


def test_kinetic_wealth_exchange_conserves_money_and_approaches_an_exponential():
    """Test that random money exchange keeps the total constant and approaches the exponential distribution."""
    trajectory = _run(boltzmann_wealth(200), 200)
    totals = {float(_agents(trajectory, 'traders', frame)['wealth'].sum()) for frame in range(201)}
    assert totals == {200.0}
    wealth = np.concatenate([_agents(trajectory, 'traders', frame)['wealth'] for frame in range(100, 201, 10)])
    for k in (1, 2, 3):
        assert np.mean(wealth >= k) == pytest.approx(0.5**k, abs=0.06)
    assert np.var(wealth) == pytest.approx(2.0, rel=0.25)


def test_toy_firm_investment_keeps_its_accounting_identities():
    """Test that every firm produces in proportion to its capital, stays solvent, and sums to aggregate production."""
    trajectory = _run(
        built := toy_firm_investment(50),
        100,
        observables=describe('toy_firm_investment').observables_of(built),
    )
    for frame in (1, 50, 100):
        firms = _agents(trajectory, 'firms', frame)
        assert np.allclose(firms['Y'], 0.1 * firms['K'])
        assert np.all(firms['A'] >= 0)
        assert np.all(firms['B'] >= 0)
        assert trajectory['production'][frame] == pytest.approx(firms['Y'].sum())


@pytest.mark.parametrize(('lifetime', 'sign'), [(10.0, -1), (15.0, 0), (20.0, 1)])
def test_capital_stock_grows_with_the_lifetime_of_capital(lifetime, sign):
    """Test that capital declines, stays, or grows exponentially as its lifetime is below, at, or above balance."""
    capital = _run(capital_stock(lifetime=lifetime), 50, dt=1.0)['economy.K']
    rate = 0.2 / 3 - 1 / lifetime
    assert capital[-1] == pytest.approx(100 * np.exp(rate * 50), rel=1e-3)
    assert np.sign(round(capital[-1] - 100, 6)) == sign


def test_exchange_rates_reproduce_the_floating_and_fixed_stories():
    """Test the stages of a 10 percent rise of home prices under a floating and a fixed exchange rate."""
    floating = _run(built := exchange_rates(), 2, dt=1, observables=describe('exchange_rates').observables_of(built))
    assert floating['home.E'][-1] == pytest.approx(0.805, abs=1e-3)
    assert floating['home.M'][-1] == pytest.approx(30.6)
    assert floating['balance'][-1] == pytest.approx(4.2, abs=0.05)
    assert floating['real_rate'][-1] == pytest.approx(floating['real_rate'][0])
    assert floating['output'][-1] == pytest.approx(103.4)
    fixed = _run(
        built := exchange_rates(floating=False),
        2,
        dt=1,
        observables=describe('exchange_rates').observables_of(built),
    )
    assert set(fixed['home.E'].round(9)) == {round(34 / (32 * 1.2), 9)}
    assert fixed['home.X'][-1] == pytest.approx(0.9 * 32)
    assert fixed['real_rate'][-1] > fixed['real_rate'][0]


def test_cournot_duopoly_converges_to_the_nash_equilibrium():
    """Test that best responses without errors converge to 50 units each at the price 500, and errors keep noise."""
    exact = _run(built := cournot_duopoly(), 30, observables=describe('cournot_duopoly').observables_of(built))
    assert exact['total'][-1] == pytest.approx(100.0, abs=1e-3)
    assert exact['price'][-1] == pytest.approx(500.0, abs=1e-2)
    assert np.allclose(_agents(exact, 'firms', 30)['output'], 50.0, atol=1e-3)
    noisy = _run(
        built := cournot_duopoly(error=0.1),
        30,
        observables=describe('cournot_duopoly').observables_of(built),
    )['total'][10:]
    assert noisy.std() > 0.5
    assert np.all(np.abs(noisy - 100.0) < 25.0)


def test_riccati_prices_converge_to_the_lower_equilibrium():
    """Test that a price below the larger root converges to the smaller root of the quadratic excess demand."""
    price = _run(riccati_prices(1.0), 10.0, dt=0.1)['market.P']
    roots = np.sort(np.roots([0.1, -3.0, 9.0]))
    assert price[-1] == pytest.approx(roots[0], rel=1e-3)


def test_produce_market_lists_shrink_until_every_shopper_is_done():
    """Test that shopping lists only shrink, and that stalls selling everything serve every list at their markups."""
    trajectory = _run(
        built := produce_market(30, 5, n_items_stocked=4, random_state=1),
        6,
        observables=describe('produce_market').observables_of(built),
    )
    assert np.all(np.diff(trajectory['items_left']) <= 0)
    assert trajectory['items_left'][-1] == 0
    complete = _run(
        built := produce_market(30, 5, n_items_stocked=12, random_state=1),
        1,
        observables=describe('produce_market').observables_of(built),
    )
    assert complete['items_left'][-1] == 0
    initial = produce_market(30, 5, n_items_stocked=12, random_state=1).elements[1].values['wanted']
    prices = (0.6, 0.5, 0.7, 0.9, 1.2, 1.5, 0.8, 0.4, 0.5, 0.3, 0.6, 1.0)
    costs = np.array([sum(price for item, price in enumerate(prices) if int(mask) >> item & 1) for mask in initial])
    ratio = _agents(complete, 'shoppers', 1)['spent'] / costs
    assert np.all((ratio >= 1.01 - 1e-9) & (ratio <= 1.3 + 1e-9))


def test_car_dealer_oscillates_with_a_short_response_delay():
    """Test that the inventory stays at 200 cars until demand steps up, then oscillates unless the response is slow."""
    fast = _run(car_dealer(), 100.0, dt=1.0)['dealer.inventory']
    slow = _run(car_dealer(response_delay=6.0), 100.0, dt=1.0)['dealer.inventory']
    assert np.allclose(fast[:26], 200.0)
    assert fast[25:].min() < 195.0
    assert np.ptp(slow[60:]) < np.ptp(fast[60:]) / 5


def test_car_wash_books_every_payment():
    """Test that the revenue prices every wash and valet, and the wash queue never exceeds the balking length."""
    trajectory = _run(car_wash(p_enter=0.05), 3600)
    till = _agents(trajectory, 'till', 3600)
    assert till['revenue'][0] == 4.0 * till['washes'][0] + 10.0 * till['valets'][0]
    assert till['washes'][0] + till['valets'][0] > 0
    assert till['turned_away'][0] > 0
    for frame in range(0, 3601, 100):
        assert np.sum(_agents(trajectory, 'cars', frame)['stage'] == 0) <= 3


def test_gibrat_process_makes_lognormal_sizes():
    """Test that the logarithm of the sizes spreads like a random walk of the logarithmic growth factors."""
    sizes = _agents(
        _run(built := gibrat_process(1000), 100, observables=describe('gibrat_process').observables_of(built)),
        'firms',
        100,
    )['x']
    logs = np.log(sizes)
    assert logs.std() == pytest.approx(np.sqrt(100) * 0.05 / 1.02, rel=0.1)
    assert abs(float(np.mean((logs - logs.mean()) ** 3)) / logs.std() ** 3) < 0.3


def test_multiplicative_process_respects_its_floor():
    """Test that no size falls below the fraction of the previous mean, and the floor fattens the upper tail."""
    trajectory = _run(
        built := multiplicative_process(300),
        100,
        observables=describe('multiplicative_process').observables_of(built),
    )
    for frame in (10, 50, 100):
        previous = _agents(trajectory, 'firms', frame)['previous']
        assert _agents(trajectory, 'firms', frame)['x'].min() >= 0.9 * previous.mean() - 1e-9
    free = _agents(
        _run(built := gibrat_process(300), 100, observables=describe('gibrat_process').observables_of(built)),
        'firms',
        100,
    )['x']
    floored = _agents(trajectory, 'firms', 100)['x']
    assert floored.max() / np.median(floored) > 1.0
    assert floored.min() / np.median(floored) > free.min() / np.median(free)


def test_kesten_process_reaches_a_positive_stationary_state():
    """Test that Kesten sizes stay positive and settle far below the initial size when the factor shrinks on average."""
    trajectory = _run(built := kesten_process(500), 100, observables=describe('kesten_process').observables_of(built))
    sizes = _agents(trajectory, 'firms', 100)['x']
    assert sizes.min() > 0
    assert np.median(sizes) < 25.0
    assert sizes.max() > 10 * np.median(sizes)


def test_bouchaud_mezard_exchange_conserves_and_equalizes_wealth():
    """Test that exchange without returns conserves the total and relaxes wealth to the mean at rate `J N / (N - 1)`."""
    wealth = np.linspace(0.5, 1.5, 20)
    trajectory = _run(
        built := bouchaud_mezard_wealth(20, sigma=0.0, wealth=wealth),
        1.0,
        dt=0.1,
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )
    final = trajectory['agents.W'][-1]
    assert final.sum() == pytest.approx(20.0)
    assert np.allclose(final - 1.0, (wealth - 1.0) * np.exp(-20 / 19), atol=2e-3)
    noisy = _run(
        built := bouchaud_mezard_wealth(20),
        1.0,
        dt=0.1,
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )['agents.W']
    assert np.all(noisy > 0)


def test_bouchaud_mezard_exchanges_with_the_mean():
    """Test that the agents exchange with the group mean as with the complete-graph Laplacian, conserving wealth."""
    wealth = np.random.default_rng(0).uniform(0.5, 1.5, 50)
    mean_field = compile_system(
        built := bouchaud_mezard_wealth(50, sigma=0.0, wealth=wealth),
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )
    complete = compile_system(
        built := bouchaud_mezard_wealth(50, graph=nx.complete_graph(50), sigma=0.0, wealth=wealth),
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )
    exchange = mean_field.drift(0.0, mean_field.initial_state, mean_field.initial_parameters)
    np.testing.assert_allclose(
        exchange,
        complete.drift(0.0, complete.initial_state, complete.initial_parameters),
        atol=1e-12,
    )
    assert abs(exchange.sum()) < 1e-12


def test_debtrank_spreads_distress_once_through_the_loans():
    """Test DebtRank on two banks by hand, and that every bank ends inactive or undistressed in a random network."""
    pair = _run(
        built := debtrank([[0.0, 5.0], [0.0, 0.0]], [10.0, 10.0]),
        5,
        observables=describe('debtrank').observables_of(built),
    )
    assert np.allclose(_agents(pair, 'banks', 5)['h'], [1.0, 0.5])
    assert pair['value_at_risk'][-1] - pair['value_at_risk'][0] == pytest.approx(0.5)
    network = _run(built := debtrank(), 10, observables=describe('debtrank').observables_of(built))
    assert np.all(np.diff(network['value_at_risk']) >= 0)
    assert network['value_at_risk'][-1] <= 1.0
    assert set(_agents(network, 'banks', 10)['status'].tolist()) <= {0.0, 2.0}


@pytest.mark.parametrize(('E', 'shift'), [(0.0, 0.0), (0.5, 1.0)])
def test_arbitrage_traders_drift_only_with_an_arbitrage_opportunity(E, shift):
    """Test that the good diffuses in place without arbitrage and drifts by `D E t^2` with it."""
    density = _run(arbitrage_traders(E=E), 2.0, dt=0.1)['market.rho'][-1]
    positions = (np.arange(200) + 0.5) * 0.1 - 10
    assert (density * positions).sum() / density.sum() == pytest.approx(shift, abs=1e-3)


def test_oil_economy_capital_rises_and_falls_with_the_resource():
    """Test that the resource only falls, and the capital peaks inside the run and then declines."""
    trajectory = _run(oil_economy(), 100.0, dt=1.0)
    assert np.all(np.diff(trajectory['oil.resource']) <= 1e-9)
    capital = trajectory['oil.capital']
    peak = int(np.argmax(capital))
    assert 10 < peak < 90
    assert capital[-1] < capital[peak] / 2


def test_fishery_capital_settles_or_collapses_with_its_technology():
    """Test that the default fleet settles at an equilibrium, and a fleet that keeps high yields depletes the stock."""
    settled = _run(fishery_capital(), 200.0, dt=1.0)
    assert np.ptp(settled['fishery.resource'][-20:]) < 1.0
    assert settled['fishery.resource'][-1] > 100.0
    efficient = tuple(np.sqrt(np.arange(11) / 10).round(3))
    collapsed = _run(fishery_capital(yield_table=efficient), 200.0, dt=1.0)
    assert collapsed['fishery.resource'][-1] < 1.0
    assert collapsed['fishery.capital'][-1] < collapsed['fishery.capital'].max() / 10


def test_nk_technology_costs_only_fall_and_fall_faster_for_modular_designs():
    """Test that the total cost never rises, and that it falls further when redesigns change fewer components."""
    modular = _run(built := nk_technology(10, 1), 2000, observables=describe('nk_technology').observables_of(built))[
        'total_cost'
    ]
    coupled = _run(built := nk_technology(10, 4), 2000, observables=describe('nk_technology').observables_of(built))[
        'total_cost'
    ]
    for costs in (modular, coupled):
        assert np.all(np.diff(costs) <= 1e-12)
    assert modular[-1] < coupled[-1]


def test_arthur_polak_invents_only_wanted_functions():
    """Test that the pool grows from the primitives and holds only correct circuits of wanted functions."""
    trajectory = _run(arthur_polak(), 1000)
    pool = _agents(trajectory, 'technologies', 1000)
    wanted = {
        (1, _truth_table(1, lambda a: not a)),
        *(
            (2, _truth_table(2, function))
            for function in (
                lambda a, b: a and b,
                lambda a, b: a or b,
                lambda a, b: a != b,
                lambda a, b: not (a or b),
                lambda a, b: a == b,
                lambda a, b: (not a) or b,
            )
        ),
        *(
            (3, _truth_table(3, function))
            for function in (
                lambda a, b, c: a and b and c,
                lambda a, b, c: a or b or c,
                lambda a, b, c: (a != b) != c,
                lambda a, b, c: (a and b) or (a and c) or (b and c),
            )
        ),
    }
    inventions = set(zip(pool['arity'][3:].astype(int).tolist(), pool['table'][3:].astype(int).tolist(), strict=True))
    assert len(inventions) == len(pool['arity']) - 3 > 0
    assert inventions <= wanted
    assert np.all(np.diff(trajectory['technologies.count']) >= 0)


@pytest.mark.parametrize(('r', 'low'), [(2.0, True), (3.0, False)])
def test_creative_phase_transition_jumps_above_the_critical_rule_density(r, low):
    """Test that the final diversity stays low below the critical rule density and jumps to almost full above it."""
    diversity = _run(creative_phase_transition(0.1, r=r), 100, dt=1)['system.x'][-1]
    assert (diversity < 0.2) if low else (diversity > 0.9)


def test_phone_adoption_spreads_among_households_that_can_afford_it():
    """Test that adoption never falls, spreads, and reaches only households above the income threshold."""
    trajectory = _run(
        built := phone_adoption((20, 20)),
        50,
        observables=describe('phone_adoption').observables_of(built),
    )
    share = trajectory['share']
    assert np.all(np.diff(share) >= 0)
    assert share[-1] > 2 * share[0]
    households = _agents(trajectory, 'households', 50)
    assert np.all(households['income'][households['phone'] == 1] >= 1.2)
    stuck = _run(
        built := phone_adoption((20, 20), threshold=100.0),
        20,
        observables=describe('phone_adoption').observables_of(built),
    )['share']
    assert set(stuck.tolist()) == {0.0}


def test_car_wash_runs_reproducibly():
    """Test that two runs of the car wash with the same random state take the same revenue."""
    first = _agents(_run(car_wash(p_enter=0.05), 600), 'till', 600)['revenue']
    second = _agents(_run(car_wash(p_enter=0.05), 600), 'till', 600)['revenue']
    assert np.array_equal(first, second)


def test_correlated_geometric_brownian_motion_correlates_returns():
    """Test that the log returns per frame of the two assets have the sample correlation 0.7 within 0.05."""
    system = correlated_geometric_brownian_motion(rho=0.7)
    trajectory = Simulation(system, solver=EulerMaruyama(dt=1e-3), dt=0.01, random_state=0).run(until=100).trajectory_
    first, second = (np.diff(np.log(trajectory[f'{label}.price'])) for label in ('first', 'second'))
    assert first.size == 10_000
    assert np.corrcoef(first, second)[0, 1] == pytest.approx(0.7, abs=0.05)
