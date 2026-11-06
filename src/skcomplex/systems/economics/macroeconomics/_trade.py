"""Macroeconomics systems: the exchange rate and the balance of trade of an open economy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import where
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class TradingEconomyElement(Element):
    """Open economy that trades with the World at an exchange rate, floating or fixed.

    Its states are the exchange rate `E` in World currency per unit of home currency, the export and import volumes
    `X` and `M`, and the home price `PH`. Its parameters are the prices, volumes, and elasticities of the balanced
    start, the rise `inflation` of home prices, whether the rate floats, and the shares of the output.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The exchange rate.

        X:
            The export volume.

        M:
            The import volume.

        PH:
            The home price of home-produced goods.

        home_price:
            The home price before the rise.

        inflation:
            The relative rise of the home prices.

        world_price:
            The world price level.

        exports:
            The export volume of the balanced start.

        imports:
            The import volume of the balanced start.

        export_price:
            The World price of the exports at the start.

        import_price:
            The home price of the imports at the start.

        export_elasticity:
            The price elasticity of the exports.

        import_elasticity:
            The price elasticity of the imports.

        floating:
            1 when the exchange rate floats, 0 when it is fixed.

        import_share:
            The share of the imports in the output.

        demand:
            The private domestic demand.

        government:
            The government spending.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import TradingEconomyElement
        >>> TradingEconomyElement('home').floating
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[object, StateSpec(domain=RealSet.open(0, oo))] = 0.885,
        X: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 32.0,
        M: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 34.0,
        PH: Annotated[object, StateSpec(domain=RealSet.open(0, oo))] = 1.2,
        home_price: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        inflation: Annotated[object, ParameterSpec()] = 0.1,
        world_price: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        exports: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 32.0,
        imports: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 34.0,
        export_price: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.062,
        import_price: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.13,
        export_elasticity: Annotated[object, ParameterSpec()] = -1.0,
        import_elasticity: Annotated[object, ParameterSpec()] = -1.0,
        floating: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        import_share: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.34,
        demand: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 52.0,
        government: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.X = X
        self.M = M
        self.PH = PH
        self.home_price = home_price
        self.inflation = inflation
        self.world_price = world_price
        self.exports = exports
        self.imports = imports
        self.export_price = export_price
        self.import_price = import_price
        self.export_elasticity = export_elasticity
        self.import_elasticity = import_elasticity
        self.floating = floating
        self.import_share = import_share
        self.demand = demand
        self.government = government


def _home_price(s: SimpleNamespace) -> Expression:
    """Return the home price after the rise."""
    home = s.target
    price: Expression = home.home_price * (1 + home.inflation)
    return price


def _exchange_rate(s: SimpleNamespace) -> Expression:
    """Return the rate that balances the currency market at the current volumes when it floats, else the rate."""
    home = s.target
    balancing = home.M * home.world_price / (home.X * home.home_price * (1 + home.inflation))
    rate: Expression = where(home.floating > 0, balancing, home.E)
    return rate


def _exports(s: SimpleNamespace) -> Expression:
    """Return the export volume, which responds to the World price of the exports with a constant elasticity."""
    home = s.target
    volume: Expression = home.exports * (1 + home.export_elasticity * (home.PH * home.E / home.export_price - 1))
    return volume


def _imports(s: SimpleNamespace) -> Expression:
    """Return the import volume, which responds to the home price of the imports with a constant elasticity."""
    home = s.target
    volume: Expression = home.imports * (
        1 + home.import_elasticity * (home.world_price / (home.E * home.import_price) - 1)
    )
    return volume


def _adjustments() -> list[MapInteraction]:
    """Return the maps of the home price, the exchange rate, and the trade volumes at every stage."""
    return [
        MapInteraction(f'{state}_adjustment', target='home', state=state, function=function)
        for state, function in (('PH', _home_price), ('E', _exchange_rate), ('X', _exports), ('M', _imports))
    ]


def _balance(s: SimpleNamespace) -> Expression:
    """Return the balance of trade of the economy `home` at home prices."""
    return s.home.X * s.home.PH - s.home.M * s.home.world_price / s.home.E


def _real_rate(s: SimpleNamespace) -> Expression:
    """Return the real exchange rate, the home price level in World currency over the world price level."""
    home = s.home
    price_level = home.world_price / home.E * home.import_share + home.PH * (1 - home.import_share)
    return price_level * home.E / home.world_price


def _output(s: SimpleNamespace) -> Expression:
    """Return the output, domestic demand and government spending plus net exports."""
    return s.home.demand + s.home.government + s.home.X - s.home.M


def _trade_observables(system: System) -> list[ObservableSpec]:
    """Return the trade balance, the real exchange rate, and the output."""
    return [
        ObservableSpec('balance', _balance),
        ObservableSpec('real_rate', _real_rate),
        ObservableSpec('output', _output),
    ]


@register_system(
    field='economics/macroeconomics',
    tags=('international trade', 'exchange rate', 'discrete map'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 2, 'dt': 1},
    observables=_trade_observables,
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
            The system, with the economy `home`, the balance of trade at home prices `balance`, the real exchange rate
            `real_rate`, and the output `output`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import exchange_rates
        >>> round(exchange_rates().elements[0].E, 3)
        0.885
    """
    rate = imports * world_price / (exports * home_price)
    home = TradingEconomyElement(
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
    return System(elements=[home], interactions=_adjustments())
