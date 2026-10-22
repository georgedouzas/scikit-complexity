"""Plantation models: the area of a tree crop that expands with its profit and collapses with its pests."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Auxiliary, Element, Law, Rate, Sink, Symbols, System, Transfer, lookup, where
from ....core.runtime import register_model

PROFITABILITY_EFFECT = (
    (0.0, 0.02),
    (0.1, 0.738),
    (0.2, 1.16),
    (0.3, 1.5),
    (0.4, 1.76),
    (0.5, 2.0),
    (0.6, 2.14),
    (0.7, 2.21),
    (0.8, 2.24),
    (0.9, 2.24),
    (1.0, 2.27),
)
YIELD_EFFECT = (
    (0.0, 0.645),
    (1.0, 0.555),
    (2.0, 0.47),
    (3.0, 0.41),
    (4.0, 0.37),
    (5.0, 0.32),
    (6.0, 0.29),
    (7.0, 0.26),
    (8.0, 0.23),
    (9.0, 0.215),
    (10.0, 0.21),
)
PROFIT_EFFECT = (
    (0.0, 1.11),
    (10.0, 1.09),
    (20.0, 1.08),
    (30.0, 1.07),
    (40.0, 1.06),
    (50.0, 1.05),
    (60.0, 1.04),
    (70.0, 1.03),
    (100.0, 1.02),
)
ECOLOGICAL_EFFECT = (
    (0.0, 1.0),
    (8.33, 0.87),
    (16.7, 0.725),
    (25.0, 0.535),
    (33.3, 0.41),
    (41.7, 0.335),
    (50.0, 0.24),
    (58.3, 0.2),
    (66.7, 0.11),
    (75.0, 0.075),
    (83.3, 0.045),
    (91.7, 0.02),
    (100.0, 0.005),
)
INTENSIFICATION_EFFECT = (
    (0.0, 1.04),
    (8.33, 1.18),
    (16.7, 1.24),
    (25.0, 1.29),
    (33.3, 1.34),
    (41.7, 1.37),
    (50.0, 1.41),
    (58.3, 1.45),
    (66.7, 1.46),
    (75.0, 1.48),
    (100.0, 1.48),
)
FULL_INDEX = 100.0


def _positive(value: Expression) -> Expression:
    """Return the positive part of a symbolic value."""
    return where(value > 0, value, 0)


def _cocoa_laws(s: Symbols) -> list[Law]:
    """Return the planting, abandonment, and conversion of cocoa land, and the thinning of shade and the insects."""
    cocoa_yield = Auxiliary(
        'cocoa_yield',
        s.yield_normal * lookup(s.insects, ECOLOGICAL_EFFECT) * lookup(s.shade_removal, INTENSIFICATION_EFFECT),
    )
    revenue = s.price * cocoa_yield.symbol
    profit = Auxiliary('profit_per_area', revenue - s.cost - s.thinning_cost * s.shade_removal / FULL_INDEX)
    profitability = Auxiliary('profitability', _positive(profit.symbol) / revenue)
    desired = s.area * lookup(profitability.symbol, PROFITABILITY_EFFECT)
    conversion = _positive(s.palm_margin - profitability.symbol) / s.palm_margin
    unprotected = 1 - s.subsidy
    return [
        cocoa_yield,
        profit,
        profitability,
        Transfer(
            s.forest,
            s.area,
            _positive(desired - s.area) / s.adjustment_time * s.forest / (s.forest + s.area),
            label='planting',
        ),
        Transfer(
            s.area,
            s.abandoned,
            s.area * lookup(cocoa_yield.symbol, YIELD_EFFECT) * lookup(profit.symbol, PROFIT_EFFECT) / s.abandon_time,
            label='abandonment',
        ),
        Transfer(s.abandoned, s.forest, (1 - conversion) * s.abandoned / s.conversion_time, label='fallow'),
        Transfer(s.abandoned, Sink('oil_palm'), conversion * s.abandoned / s.conversion_time, label='conversion'),
        Rate(s.shade_removal, unprotected * (FULL_INDEX - s.shade_removal) / s.thinning_time),
        Rate(
            s.insects,
            s.insect_growth * unprotected * s.shade_removal / FULL_INDEX * s.insects * (1 - s.insects / FULL_INDEX)
            - s.insect_control * s.subsidy * s.insects,
        ),
    ]


Plantation = Element.define(
    'Plantation',
    states={
        'forest': state(default=490.0, domain=Domain.NONNEGATIVE, quantity='land'),
        'area': state(default=10.0, domain=Domain.NONNEGATIVE, quantity='land'),
        'abandoned': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='land'),
        'shade_removal': state(default=5.0, domain=Domain.NONNEGATIVE),
        'insects': state(default=1.0, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'yield_normal': parameter(default=5.0, domain=Domain.NONNEGATIVE),
        'price': parameter(default=40.0, domain=Domain.NONNEGATIVE),
        'cost': parameter(default=100.0, domain=Domain.NONNEGATIVE),
        'thinning_cost': parameter(default=20.0, domain=Domain.NONNEGATIVE),
        'adjustment_time': parameter(default=3.0, domain=Domain.POSITIVE),
        'abandon_time': parameter(default=4.0, domain=Domain.POSITIVE),
        'conversion_time': parameter(default=3.0, domain=Domain.POSITIVE),
        'palm_margin': parameter(default=0.4, domain=Domain.POSITIVE),
        'thinning_time': parameter(default=8.0, domain=Domain.POSITIVE),
        'insect_growth': parameter(default=0.6, domain=Domain.NONNEGATIVE),
        'insect_control': parameter(default=0.3, domain=Domain.NONNEGATIVE),
        'subsidy': parameter(default=0.0, domain=Domain.NONNEGATIVE),
    },
    laws=_cocoa_laws,
)


@register_model(
    field='economics/resources',
    tags=('agriculture', 'boom and bust', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 50.0, 'dt': 0.25},
)
def cocoa(
    area: float = 10.0,
    *,
    price: float = 40.0,
    cost: float = 100.0,
    insect_growth: float = 0.6,
    subsidy: float = 0.0,
) -> System:
    """Return a cocoa production system, plantations that boom with their profit and bust with the insects they invite.

    The planted area moves towards a desired area, the current area times a table function of the profit margin, so
    a profitable crop expands it. Farmers thin the shading trees, which raises the yield through an intensification
    table but lets the insects multiply. Insect attack lowers the yield through an ecological table, and the area is
    abandoned at a rate that falls with the yield and the profit through two more tables. Abandoned land is converted
    to oil palm. A subsidy for shade conservation and pest control slows the thinning and controls the insects. Land
    is conserved, so the area booms while the yield is high and busts once the insects take over, unless a subsidy
    keeps the shade. The structure and the five table functions follow the boom and bust of cocoa in Malaysia of
    Bala, Arshad and Noh (2017, Chapter 8), and the rates, the price, and the costs are illustrative, with the area in
    thousands of hectares, the yield in hundreds of kilograms per hectare, the money in ringgit per hundred
    kilograms and per hectare, the shade removal and the insect attack as indices from 0 to 100, and time in years.

    Args:
        area:
            The initial planted area.

        price:
            The farm price of cocoa.

        cost:
            The operating cost per hectare.

        insect_growth:
            The largest relative growth rate of the insect attack, reached with all shade removed.

        subsidy:
            The coverage of the subsidy and extension programme, from 0 to 1.

    Returns:
        system:
            The model, with the land stocks `cocoa.area` and `cocoa.abandoned`, the indices `cocoa.shade_removal` and
            `cocoa.insects`, the auxiliaries `cocoa.cocoa_yield`, `cocoa.profit_per_area`, and `cocoa.profitability`,
            and the flows `cocoa.planting`, `cocoa.abandonment`, and `cocoa.conversion` per year, over years.

    Examples:
        >>> from skcomplex.models.economics.resources import cocoa
        >>> cocoa(subsidy=1.0).elements[0].subsidy
        1.0
    """
    plantation = Plantation(
        'cocoa',
        area=area,
        price=price,
        cost=cost,
        insect_growth=insect_growth,
        subsidy=subsidy,
    )
    return System(components=[plantation])
