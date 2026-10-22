"""Macroeconomics: growth, business cycles, investment, international trade, and mortal monetary economies."""

from ._banking import Depositor, Regulator, ReserveBank, fractional_reserve_banking
from ._capital import CapitalStock, capital_stock
from ._credit import Bank, CreditMarket, Firm, credit_networks
from ._firms import InvestingFirm, toy_firm_investment
from ._growth import Economy, Income, SolowGrowth, multiplier_accelerator, solow
from ._metaphoria import (
    Business,
    Census,
    CentralBank,
    Commons,
    FoodMarket,
    LabourMarket,
    Plot,
    Settler,
    metaphoria,
)
from ._monetary import Circuit, Firms, Households, MonetaryBank, stock_flow_consistent_economy
from ._trade import TradingEconomy, exchange_rates

__all__: list[str] = [
    'Bank',
    'Business',
    'CapitalStock',
    'Census',
    'CentralBank',
    'Circuit',
    'Commons',
    'CreditMarket',
    'Depositor',
    'Economy',
    'Firm',
    'Firms',
    'FoodMarket',
    'Households',
    'Income',
    'InvestingFirm',
    'LabourMarket',
    'MonetaryBank',
    'Plot',
    'Regulator',
    'ReserveBank',
    'Settler',
    'SolowGrowth',
    'TradingEconomy',
    'capital_stock',
    'credit_networks',
    'exchange_rates',
    'fractional_reserve_banking',
    'metaphoria',
    'multiplier_accelerator',
    'solow',
    'stock_flow_consistent_economy',
    'toy_firm_investment',
]
