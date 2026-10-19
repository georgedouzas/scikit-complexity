"""Macroeconomics models: the exchange rate and the balance of trade of an open economy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Observable, Symbols, System, Update, where
from ....core.runtime import register_model

TradingEconomy = Element.define(
    'TradingEconomy',
    states={
        'E': state(default=0.885, domain=Domain.POSITIVE),
        'X': state(default=32.0, domain=Domain.NONNEGATIVE),
        'M': state(default=34.0, domain=Domain.NONNEGATIVE),
        'PH': state(default=1.2, domain=Domain.POSITIVE),
    },
    parameters={
        'home_price': parameter(default=1.2, domain=Domain.POSITIVE),
        'inflation': parameter(default=0.1, domain=Domain.REAL),
        'world_price': parameter(default=1.0, domain=Domain.POSITIVE),
        'exports': parameter(default=32.0, domain=Domain.NONNEGATIVE),
        'imports': parameter(default=34.0, domain=Domain.NONNEGATIVE),
        'export_price': parameter(default=1.062, domain=Domain.POSITIVE),
        'import_price': parameter(default=1.13, domain=Domain.POSITIVE),
        'export_elasticity': parameter(default=-1.0, domain=Domain.REAL),
        'import_elasticity': parameter(default=-1.0, domain=Domain.REAL),
        'floating': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'import_share': parameter(default=0.34, domain=Domain.NONNEGATIVE),
        'demand': parameter(default=52.0, domain=Domain.NONNEGATIVE),
        'government': parameter(default=50.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Update(s.PH, s.home_price * (1 + s.inflation)),
        Update(s.E, where(s.floating > 0, s.M * s.world_price / (s.X * s.home_price * (1 + s.inflation)), s.E)),
        Update(s.X, s.exports * (1 + s.export_elasticity * (s.PH * s.E / s.export_price - 1))),
        Update(s.M, s.imports * (1 + s.import_elasticity * (s.world_price / (s.E * s.import_price) - 1))),
    ],
)


def _balance(s: Symbols) -> Expression:
    """Return the balance of trade of the economy `home` at home prices."""
    return s.home.X * s.home.PH - s.home.M * s.home.world_price / s.home.E


def _real_rate(s: Symbols) -> Expression:
    """Return the real exchange rate, the home price level in World currency over the world price level."""
    home = s.home
    price_level = home.world_price / home.E * home.import_share + home.PH * (1 - home.import_share)
    return price_level * home.E / home.world_price


def _output(s: Symbols) -> Expression:
    """Return the output, domestic demand and government spending plus net exports."""
    return s.home.demand + s.home.government + s.home.X - s.home.M


@register_model(
    field='economics/macroeconomics',
    tags=('international trade', 'exchange rate', 'discrete map'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 2, 'dt': 1},
)
def exchange_rates(
    exports: float = 32.0,
    imports: float = 34.0,
    *,
    home_price: float = 1.2,
    inflation: float = 0.1,
    export_elasticity: float = -1.0,
    import_elasticity: float = -1.0,
    floating: bool = True,
    demand: float = 52.0,
    government: float = 50.0,
    world_price: float = 1.0,
) -> System:
    """Return Hamill and Gilbert's staged adjustment of trade and the exchange rate to a rise in home prices.

    The economy starts in balance at the exchange rate `E = M * PW / (X * PH)`, in units of World currency per unit of
    Home currency, with the export and import volumes and prices of the United Kingdom in 1999 by default. Home prices
    rise by `inflation` at the first stage. At every stage a floating rate takes the value that balances the currency
    market at the current volumes, a fixed rate stays, and the volumes respond to the prices of the stage before with
    constant price elasticities. The book follows two stages, the default.

    Args:
        exports:
            The initial export volume.

        imports:
            The initial import volume.

        home_price:
            The home price of home-produced goods before the rise.

        inflation:
            The relative rise of the home prices.

        export_elasticity:
            The price elasticity of the exports.

        import_elasticity:
            The price elasticity of the imports.

        floating:
            Whether the exchange rate floats, or is fixed as in the Eurozone.

        demand:
            The private domestic demand, consumption and investment.

        government:
            The government spending.

        world_price:
            The world price level.

    Returns:
        system:
            The model, with the economy `home`, the balance of trade at home prices `balance`, the real exchange rate
            `real_rate`, and the output `output`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import exchange_rates
        >>> round(exchange_rates().elements[0].E, 3)
        0.885
    """
    rate = imports * world_price / (exports * home_price)
    home = TradingEconomy(
        'home',
        E=rate,
        X=exports,
        M=imports,
        PH=home_price,
        home_price=home_price,
        inflation=inflation,
        world_price=world_price,
        exports=exports,
        imports=imports,
        export_price=home_price * rate,
        import_price=world_price / rate,
        export_elasticity=export_elasticity,
        import_elasticity=import_elasticity,
        floating=float(floating),
        import_share=imports / (demand + government + exports - imports),
        demand=demand,
        government=government,
    )
    observables = [Observable('balance', _balance), Observable('real_rate', _real_rate), Observable('output', _output)]
    return System(elements=[home], observables=observables)
