"""Test the `_shops` module of `skcomplex.models.economics.markets`."""

import numpy as np
from skcomplex.core.components import Element

from skcomplex.core.base import parameter, state
from skcomplex.core.elements import GroupElement
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models import run
from skcomplex.models.economics.realizations import (
    InstitutionInteraction,
)


class ShopElement(Element):
    """Shop whose price grows by a drift and noise, slowed by the interest rate."""

    price: float = state(1.0)
    drift: float = parameter(0.03)

    def laws(self, s):
        """Return the pricing rule."""
        return [RuleLaw(self.inflate)]

    def inflate(self, agent, model):
        """Raise the price."""
        agent.price *= 1 + model.parameter('drift') + model.rng.normal(0, 0.02) - model.parameters['bank.rate']


class CentralBankInteraction(InstitutionInteraction):
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
        elements=[GroupElement('shops', element=ShopElement, size=20)],
        interactions=[CentralBankInteraction('bank', phase='policy')],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('policy')])),
    )
    observables = [ObservableSpec('price_level', group='shops', variable='price')]
    trajectory = Simulation(system, random_state=0, observables=observables).run(until=40).trajectory_
    rate, inflation = trajectory.values('bank.rate'), trajectory.values('bank.inflation')
    changed = np.diff(rate) != 0
    assert changed.any()
    assert not changed.all()
    np.testing.assert_array_equal(changed, inflation[1:] > 0.02)
    assert np.isnan(inflation[0])


def test_digital_market_closes_shops_and_lowers_prices():
    """Test that shopping online leaves fewer shops, a larger leader, and lower prices than shopping in person."""
    local = run('shops', online=0.0, settings={'until': 100}).trajectory_
    digital = run('shops', online=1.0, settings={'until': 100}).trajectory_
    assert digital.values('open_shops')[-1] < local.values('open_shops')[-1]
    assert digital.values('concentration')[-1] > 2 * local.values('concentration')[-1]
    assert digital.values('mean_price')[50:].mean() < local.values('mean_price')[50:].mean()
