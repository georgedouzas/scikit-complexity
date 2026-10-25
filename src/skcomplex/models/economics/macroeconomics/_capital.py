"""Macroeconomics models: physical capital that reinvests its output and depreciates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import NumericDomain, parameter, state
from ....core.components import ElementComponent
from ....core.laws import RateLaw
from ....core.system import System
from ....engine.runtime import register_model

CapitalStockComponent = ElementComponent.define(
    'CapitalStockComponent',
    states={'K': state(default=100.0, domain=NumericDomain.NONNEGATIVE)},
    parameters={
        'investment_fraction': parameter(default=0.2, domain=NumericDomain.NONNEGATIVE),
        'output_per_capital': parameter(default=1 / 3, domain=NumericDomain.POSITIVE),
        'lifetime': parameter(default=15.0, domain=NumericDomain.POSITIVE),
    },
    laws=lambda s: [RateLaw(s.K, s.investment_fraction * s.output_per_capital * s.K - s.K / s.lifetime)],
)


@register_model(
    field='economics/macroeconomics',
    tags=('growth', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 50.0, 'dt': 1.0},
)
def capital_stock(
    capital: float = 100.0,
    *,
    investment_fraction: float = 0.2,
    output_per_capital: float = 1 / 3,
    lifetime: float = 15.0,
) -> System:
    """Return Meadows' capital stock, grown by reinvested output and drained by depreciation.

    The stock follows `dK/dt = investment_fraction * output_per_capital * K - K / lifetime`. It grows exponentially when
    the capital lives longer than `1 / (investment_fraction * output_per_capital)`, declines when it lives shorter, and
    stays in dynamic equilibrium otherwise.

    Args:
        capital:
            The initial capital stock.

        investment_fraction:
            The fraction of the output that is reinvested.

        output_per_capital:
            The annual output per unit of capital.

        lifetime:
            The average lifetime of the capital.

    Returns:
        system:
            The model, with the capital stock `economy.K`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import capital_stock
        >>> capital_stock(lifetime=20.0).elements[0].lifetime
        20.0
    """
    economy = CapitalStockComponent(
        'economy',
        K=capital,
        investment_fraction=investment_fraction,
        output_per_capital=output_per_capital,
        lifetime=lifetime,
    )
    return System(components=[economy])
