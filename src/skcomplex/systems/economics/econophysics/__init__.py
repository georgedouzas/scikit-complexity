"""Econophysics: wealth exchange between agents."""

from ._coalitions import CoalitionBarInteraction, CoalitionPatronElement, el_farol_coalitions
from ._crowd_avoidance import CrowdAvoidanceInteraction
from ._debt_exchange import DebtExchangeInteraction
from ._diner import DinerElement
from ._el_farol import el_farol
from ._exchange import TraderElement, boltzmann_wealth
from ._kinetic import kinetic_wealth_exchange
from ._kolkata import kolkata_restaurant
from ._minority import minority_game
from ._minority_game import MinorityGameInteraction, strategy_table
from ._minority_player import MinorityPlayerElement
from ._patron import PatronElement
from ._predictor_pool import PredictorPoolElement, forecasts
from ._restaurants import RestaurantsInteraction
from ._saving_exchange import SavingExchangeInteraction
from ._saving_trader import SavingTraderElement
from ._wealth_exchange import WealthExchangeInteraction, exchange_table
from ._welfare import (
    gini_coefficient,
    kinetic_wealth_welfare,
    mean_wealth,
    wealth_classes,
    wealth_density,
    wealth_observables,
)
from ._welfare_policy import WelfarePolicyInteraction, policy_table

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
