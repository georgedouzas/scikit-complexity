"""Econophysics: wealth exchange between agents."""

from ._el_farol import Bar, Patron, el_farol, forecasts
from ._exchange import Trader, boltzmann_wealth
from ._kinetic import DebtExchange, Saver, SavingExchange, kinetic_wealth_exchange
from ._kolkata import CrowdAvoidingDiner, Diner, Restaurants, kolkata_restaurant
from ._minority import MinorityGame, Speculator, minority_game, strategy_table

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
    'boltzmann_wealth',
    'el_farol',
    'forecasts',
    'kinetic_wealth_exchange',
    'kolkata_restaurant',
    'minority_game',
    'strategy_table',
]
