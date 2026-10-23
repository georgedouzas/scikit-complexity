"""Test the `_shops` module of `skcomplex.models.economics.markets`."""

import numpy as np

from skcomplex.core.base import parameter, state
from skcomplex.core.components import Element
from skcomplex.core.formalisms.agents import Phase, Phases, Rule
from skcomplex.core.runtime import Simulation
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import Observable, System
from skcomplex.models import run
from skcomplex.models.economics.realizations import (
    Institution,
)


class Shop(Element):
    """Shop whose price grows by a drift and noise, slowed by the interest rate."""

    price: float = state(1.0)
    drift: float = parameter(0.03)

    def laws(self, s):
        """Return the pricing rule."""
        return [Rule(self.inflate)]

    def inflate(self, agent, model):
        """Raise the price."""
        agent.price *= 1 + model.parameter('drift') + model.rng.normal(0, 0.02) - model.parameters['bank.rate']


class CentralBank(Institution):
    """Central bank that raises its rate whenever inflation of the price level exceeds the target."""

    reports = ('inflation',)
    rate: float = parameter(0.0)
    target: float = parameter(0.02)

    def act(self, model):
        """Measure inflation and raise the rate above the target."""
        level = model.observable('price_level')
        inflation = level / model.storage.get('level', 1.0) - 1
        model.storage['level'] = level
        model.report('inflation', inflation)
        if inflation > model.parameter('target'):
            model.set_parameter('rate', model.parameter('rate') + 0.005)


def test_interest_rate_rule():
    """The interest rate changes only at steps where inflation exceeds the target."""
    system = System(
        components=[Shop.group('shops', 20)],
        interactions=[CentralBank('bank', phase='policy')],
        time=DiscreteTime(schedule=Phases([Phase('policy')])),
    )
    observables = [Observable('price_level', group='shops', variable='price')]
    trajectory = Simulation(system, random_state=0, observables=observables).run(until=40).trajectory_
    rate, inflation = trajectory['bank.rate'], trajectory['bank.inflation']
    changed = np.diff(rate) != 0
    assert changed.any()
    assert not changed.all()
    np.testing.assert_array_equal(changed, inflation[1:] > 0.02)
    assert np.isnan(inflation[0])


def test_digital_market_closes_shops_and_lowers_prices():
    """Test that shopping online leaves fewer shops, a larger leader, and lower prices than shopping in person."""
    local = run('shops', online=0.0, settings={'until': 100}).trajectory_
    digital = run('shops', online=1.0, settings={'until': 100}).trajectory_
    assert digital['open_shops'][-1] < local['open_shops'][-1]
    assert digital['concentration'][-1] > 2 * local['concentration'][-1]
    assert digital['mean_price'][50:].mean() < local['mean_price'][50:].mean()
