"""Econophysics: wealth exchange between agents."""

from ._coalitions import CoalitionBarInteraction, CoalitionPatronElement, el_farol_coalitions
from ._el_farol import PatronElement, PredictorPoolElement, el_farol, forecasts
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
    'CoalitionBarInteraction',
    'CoalitionPatronElement',
    'CrowdAvoidanceInteraction',
    'DebtExchangeInteraction',
    'DinerElement',
    'MinorityGameInteraction',
    'MinorityPlayerElement',
    'PatronElement',
    'PredictorPoolElement',
    'RestaurantsInteraction',
    'SavingExchangeInteraction',
    'SavingTraderElement',
    'TraderElement',
    'WealthExchangeInteraction',
    'WelfarePolicyInteraction',
    'boltzmann_wealth',
    'el_farol',
    'el_farol_coalitions',
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
