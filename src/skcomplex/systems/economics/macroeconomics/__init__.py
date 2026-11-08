"""Macroeconomics: growth, business cycles, investment, and trade.

It also holds international trade and mortal monetary economies.
"""

from ._banking import DepositorElement, FractionalReserveBankingSystem, PolicySwitchInteraction, ReserveBankElement
from ._capital import CapitalStockElement, CapitalStockSystem
from ._credit import CreditMarketInteraction, CreditNetworksSystem, FirmElement, LenderElement
from ._firms import InvestingFirmElement, ToyFirmInvestmentSystem
from ._growth import EconomyElement, IncomeElement, MultiplierAcceleratorSystem, SolowGrowthInteraction, SolowSystem
from ._herding import HerdingAgentElement, NetworkHerdingSystem
from ._mark0 import Mark0EconomyInteraction, Mark0FirmElement, Mark0System
from ._metaphoria import (
    BusinessElement,
    CentralBankElement,
    CommonsInteraction,
    EstateInteraction,
    FoodMarketInteraction,
    LabourMarketInteraction,
    MetaphoriaSystem,
    PlotElement,
    SettlerElement,
)
from ._monetary import (
    CircuitInteraction,
    FirmsElement,
    HouseholdsElement,
    MonetaryBankElement,
    StockFlowConsistentEconomySystem,
)
from ._optimal_growth import OptimalConsumptionSystem, RamseyEconomyElement
from ._purchasing_power import CategorizedPurchasingPowerSystem, PurchasingPowerElement
from ._stochastic_growth import BrockMirmanGrowthSystem, StochasticGrowthElement
from ._trade import ExchangeRatesSystem, TradingEconomyElement

__all__: list[str] = [
    'BrockMirmanGrowthSystem',
    'BusinessElement',
    'CapitalStockElement',
    'CapitalStockSystem',
    'CategorizedPurchasingPowerSystem',
    'CentralBankElement',
    'CircuitInteraction',
    'CommonsInteraction',
    'CreditMarketInteraction',
    'CreditNetworksSystem',
    'DepositorElement',
    'EconomyElement',
    'EstateInteraction',
    'ExchangeRatesSystem',
    'FirmElement',
    'FirmsElement',
    'FoodMarketInteraction',
    'FractionalReserveBankingSystem',
    'HerdingAgentElement',
    'HouseholdsElement',
    'IncomeElement',
    'InvestingFirmElement',
    'LabourMarketInteraction',
    'LenderElement',
    'Mark0EconomyInteraction',
    'Mark0FirmElement',
    'Mark0System',
    'MetaphoriaSystem',
    'MonetaryBankElement',
    'MultiplierAcceleratorSystem',
    'NetworkHerdingSystem',
    'OptimalConsumptionSystem',
    'PlotElement',
    'PolicySwitchInteraction',
    'PurchasingPowerElement',
    'RamseyEconomyElement',
    'ReserveBankElement',
    'SettlerElement',
    'SolowGrowthInteraction',
    'SolowSystem',
    'StochasticGrowthElement',
    'StockFlowConsistentEconomySystem',
    'ToyFirmInvestmentSystem',
    'TradingEconomyElement',
]
