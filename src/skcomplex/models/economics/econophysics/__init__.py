"""Econophysics: wealth exchange between agents."""

from ._el_farol import BarInteraction, PatronElement, el_farol, forecasts
from ._exchange import TraderElement, boltzmann_wealth
from ._kinetic import DebtExchangeInteraction, SavingExchangeInteraction, SavingTraderElement, kinetic_wealth_exchange
from ._kolkata import CrowdAvoidanceInteraction, DinerElement, RestaurantsInteraction, kolkata_restaurant
from ._minority import MinorityGameInteraction, MinorityPlayerElement, minority_game, strategy_table
from ._welfare import (
    WealthExchangeInteraction,
    WelfarePolicyInteraction,
    exchange_table,
    gini_coefficient,
    kinetic_wealth_welfare,
    mean_wealth,
    policy_table,
    wealth_classes,
    wealth_density,
    wealth_observables,
)

__all__: list[str] = [
    'BarInteraction',
    'CrowdAvoidanceInteraction',
    'DebtExchangeInteraction',
    'DinerElement',
    'MinorityGameInteraction',
    'MinorityPlayerElement',
    'PatronElement',
    'RestaurantsInteraction',
    'SavingExchangeInteraction',
    'SavingTraderElement',
    'TraderElement',
    'WealthExchangeInteraction',
    'WelfarePolicyInteraction',
    'boltzmann_wealth',
    'el_farol',
    'exchange_table',
    'forecasts',
    'gini_coefficient',
    'kinetic_wealth_exchange',
    'kinetic_wealth_welfare',
    'kolkata_restaurant',
    'mean_wealth',
    'minority_game',
    'policy_table',
    'strategy_table',
    'wealth_classes',
    'wealth_density',
    'wealth_observables',
]
