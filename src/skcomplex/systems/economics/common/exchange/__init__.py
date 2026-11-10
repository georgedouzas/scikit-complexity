"""Exchange of goods shared by the economics packages.

They are Cobb-Douglas utilities, marginal rates of substitution, and trades of goods between agents.
"""

from ._exchange import cobb_douglas, marginal_rate, trade_goods
from ._exchange_interaction import ExchangeInteraction

__all__: list[str] = [
    'ExchangeInteraction',
    'cobb_douglas',
    'marginal_rate',
    'trade_goods',
]
