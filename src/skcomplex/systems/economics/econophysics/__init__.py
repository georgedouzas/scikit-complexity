"""Econophysics: wealth exchange between agents."""

from ._coalitions import CoalitionBarInteraction, CoalitionPatronElement, ElFarolCoalitionsSystem
from ._el_farol import ElFarolSystem, PatronElement, PredictorPoolElement, forecasts
from ._exchange import BoltzmannWealthSystem, TraderElement
from ._kinetic import (
    DebtExchangeInteraction,
    KineticWealthExchangeSystem,
    SavingExchangeInteraction,
    SavingTraderElement,
)
from ._kolkata import CrowdAvoidanceInteraction, DinerElement, KolkataRestaurantSystem, RestaurantsInteraction
from ._minority import MinorityGameInteraction, MinorityGameSystem, MinorityPlayerElement, strategy_table
from ._welfare import (
    KineticWealthWelfareSystem,
    WealthExchangeInteraction,
    WelfarePolicyInteraction,
    exchange_table,
    gini_coefficient,
    mean_wealth,
    policy_table,
    wealth_classes,
    wealth_density,
    wealth_observables,
)

__all__: list[str] = [
    'BoltzmannWealthSystem',
    'CoalitionBarInteraction',
    'CoalitionPatronElement',
    'CrowdAvoidanceInteraction',
    'DebtExchangeInteraction',
    'DinerElement',
    'ElFarolCoalitionsSystem',
    'ElFarolSystem',
    'KineticWealthExchangeSystem',
    'KineticWealthWelfareSystem',
    'KolkataRestaurantSystem',
    'MinorityGameInteraction',
    'MinorityGameSystem',
    'MinorityPlayerElement',
    'PatronElement',
    'PredictorPoolElement',
    'RestaurantsInteraction',
    'SavingExchangeInteraction',
    'SavingTraderElement',
    'TraderElement',
    'WealthExchangeInteraction',
    'WelfarePolicyInteraction',
    'exchange_table',
    'forecasts',
    'gini_coefficient',
    'mean_wealth',
    'policy_table',
    'strategy_table',
    'wealth_classes',
    'wealth_density',
    'wealth_observables',
]
