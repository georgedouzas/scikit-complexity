"""Macroeconomics: growth, business cycles, investment, international trade, and mortal monetary economies."""

from ._banking import DepositorElement, PolicySwitchInteraction, ReserveBankElement, fractional_reserve_banking
from ._capital import CapitalStockElement, capital_stock
from ._credit import CreditMarketInteraction, FirmElement, LenderElement, credit_networks
from ._firms import InvestingFirmElement, toy_firm_investment
from ._growth import EconomyElement, IncomeElement, SolowGrowthInteraction, multiplier_accelerator, solow
from ._metaphoria import (
    BusinessElement,
    CentralBankElement,
    CommonsInteraction,
    EstateInteraction,
    FoodMarketInteraction,
    LabourMarketInteraction,
    PlotElement,
    SettlerElement,
    metaphoria,
)
from ._monetary import (
    CircuitInteraction,
    FirmsElement,
    HouseholdsElement,
    MonetaryBankElement,
    stock_flow_consistent_economy,
)
from ._optimal_growth import RamseyEconomyElement, optimal_consumption
from ._trade import TradingEconomyElement, exchange_rates

__all__: list[str] = [
    'BusinessElement',
    'CapitalStockElement',
    'CentralBankElement',
    'CircuitInteraction',
    'CommonsInteraction',
    'CreditMarketInteraction',
    'DepositorElement',
    'EconomyElement',
    'EstateInteraction',
    'FirmElement',
    'FirmsElement',
    'FoodMarketInteraction',
    'HouseholdsElement',
    'IncomeElement',
    'InvestingFirmElement',
    'LabourMarketInteraction',
    'LenderElement',
    'MonetaryBankElement',
    'PlotElement',
    'PolicySwitchInteraction',
    'RamseyEconomyElement',
    'ReserveBankElement',
    'SettlerElement',
    'SolowGrowthInteraction',
    'TradingEconomyElement',
    'capital_stock',
    'credit_networks',
    'exchange_rates',
    'fractional_reserve_banking',
    'metaphoria',
    'multiplier_accelerator',
    'optimal_consumption',
    'solow',
    'stock_flow_consistent_economy',
    'toy_firm_investment',
]
