"""Econophysics: wealth exchange between agents."""

from ._el_farol import Bar, Patron, el_farol, forecasts
from ._exchange import Trader, boltzmann_wealth
from ._kinetic import DebtExchange, Saver, SavingExchange, kinetic_wealth_exchange
from ._kolkata import CrowdAvoidingDiner, Diner, Restaurants, kolkata_restaurant
from ._minority import MinorityGame, Speculator, minority_game, strategy_table
from ._welfare import (
    WealthClass,
    WealthExchange,
    WelfarePolicy,
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
    'Bar',
    'CrowdAvoidingDiner',
    'DebtExchange',
    'Diner',
    'MinorityGame',
    'Patron',
    'Restaurants',
    'Saver',
    'SavingExchange',
    'Speculator',
    'Trader',
    'WealthClass',
    'WealthExchange',
    'WelfarePolicy',
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
