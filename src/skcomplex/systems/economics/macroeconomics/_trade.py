"""Macroeconomics systems: the exchange rate and the balance of trade.

The economy is open.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System


class TradingEconomyElement(Element):
    """Open economy that trades with the World at an exchange rate, floating or fixed.

    Its states are the exchange rate `E` in World currency per unit of home currency, the export and import volumes
    `X` and `M`, and the home price `PH`. Its parameters are the prices, volumes, and elasticities of the balanced
    start, the rise `inflation` of home prices, whether the rate floats, and the shares of the output.

    Args:
        label:
            The label of the economy.

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
        *,
        E: Annotated[object, RealSet.open(0, oo)] = 0.885,
        X: Annotated[object, RealSet.closed_open(0, oo)] = 32.0,
        M: Annotated[object, RealSet.closed_open(0, oo)] = 34.0,
        PH: Annotated[object, RealSet.open(0, oo)] = 1.2,
        home_price: Annotated[object, RealSet.open(0, oo)] = 1.2,
        inflation: object = 0.1,
        world_price: Annotated[object, RealSet.open(0, oo)] = 1.0,
        exports: Annotated[object, RealSet.closed_open(0, oo)] = 32.0,
        imports: Annotated[object, RealSet.closed_open(0, oo)] = 34.0,
        export_price: Annotated[object, RealSet.open(0, oo)] = 1.062,
        import_price: Annotated[object, RealSet.open(0, oo)] = 1.13,
        export_elasticity: object = -1.0,
        import_elasticity: object = -1.0,
        floating: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        import_share: Annotated[object, RealSet.closed_open(0, oo)] = 0.34,
        demand: Annotated[object, RealSet.closed_open(0, oo)] = 52.0,
        government: Annotated[object, RealSet.closed_open(0, oo)] = 50.0,
    ) -> None:
        super().__init__(label)
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


def _balance(s: SimpleNamespace) -> Expression:
    """Return the balance of trade of the economy `home` at home prices."""
    return s.home.X * s.home.PH - s.home.M * s.home.world_price / s.home.E


def _get_real_rate(s: SimpleNamespace) -> Expression:
    """Return the real exchange rate.

    It is the home price level in World currency over the world price level.
    """
    home = s.home
    price_level = home.world_price / home.E * home.import_share + home.PH * (1 - home.import_share)
    return price_level * home.E / home.world_price


def _get_output(s: SimpleNamespace) -> Expression:
    """Return the output, domestic demand and government spending plus net exports."""
    return s.home.demand + s.home.government + s.home.X - s.home.M


class ExchangeRatesSystem(BaseSystem):
    """Hamill and Gilbert's staged adjustment of trade and the exchange rate.

    The adjustment follows a rise in home prices. The economy starts in balance at the exchange rate
    `E = M * PW / (X * PH)`, in units of World currency per unit of Home currency, with the export and import volumes
    and prices of the United Kingdom in 1999 by default. Home prices rise by `inflation` at the first stage. At every
    stage a floating rate takes the value that balances the currency market at the current volumes, a fixed rate stays,
    and the volumes respond to the prices of the stage before with constant price elasticities. The book follows two
    stages, the default.

    The system has the economy `home`, the balance of trade at home prices `balance`, the real exchange rate
    `real_rate`, and the output `output`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import ExchangeRatesSystem
        >>> round(ExchangeRatesSystem().prepare().system_.elements[0].E, 3)
        0.885
    """

    field = 'economics/macroeconomics'
    topics = ('international trade', 'exchange rate', 'discrete map')
    references = ('Hamill and Gilbert 2016',)
    until = 2
    dt = 1
    observables = (
        Observable('balance', _balance),
        Observable('real_rate', _get_real_rate),
        Observable('output', _get_output),
    )

    @staticmethod
    def _get_home_price(s: SimpleNamespace) -> Expression:
        """Return the home price after the rise."""
        home = s.target
        price: Expression = home.home_price * (1 + home.inflation)
        return price

    @staticmethod
    def _exchange_rate(s: SimpleNamespace) -> Expression:
        """Return the rate that balances the currency market when it floats.

        It balances the market at the current volumes, and otherwise it is the rate.
        """
        home = s.target
        balancing = home.M * home.world_price / (home.X * home.home_price * (1 + home.inflation))
        rate: Expression = where(home.floating > 0, balancing, home.E)
        return rate

    @staticmethod
    def _get_exports(s: SimpleNamespace) -> Expression:
        """Return the export volume.

        It responds to the World price of the exports with a constant elasticity.
        """
        home = s.target
        volume: Expression = home.exports * (1 + home.export_elasticity * (home.PH * home.E / home.export_price - 1))
        return volume

    @staticmethod
    def _get_imports(s: SimpleNamespace) -> Expression:
        """Return the import volume.

        It responds to the home price of the imports with a constant elasticity.
        """
        home = s.target
        volume: Expression = home.imports * (
            1 + home.import_elasticity * (home.world_price / (home.E * home.import_price) - 1)
        )
        return volume

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.exports = exports
        self.imports = imports
        self.home_price = home_price
        self.inflation = inflation
        self.export_elasticity = export_elasticity
        self.import_elasticity = import_elasticity
        self.floating = floating
        self.demand = demand
        self.government = government
        self.world_price = world_price

    @staticmethod
    def _get_adjustments() -> list[Interaction]:
        """Return the maps of the home price, the exchange rate, and the trade volumes.

        There is one at every stage.
        """
        system = ExchangeRatesSystem
        return [
            Interaction(
                'PH_adjustment',
                roles={'target': 'home'},
                laws=lambda s: [UpdateLaw(s.target.PH, system._get_home_price(s))],
            ),
            Interaction(
                'E_adjustment',
                roles={'target': 'home'},
                laws=lambda s: [UpdateLaw(s.target.E, system._exchange_rate(s))],
            ),
            Interaction(
                'X_adjustment',
                roles={'target': 'home'},
                laws=lambda s: [UpdateLaw(s.target.X, system._get_exports(s))],
            ),
            Interaction(
                'M_adjustment',
                roles={'target': 'home'},
                laws=lambda s: [UpdateLaw(s.target.M, system._get_imports(s))],
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rate = self.imports * self.world_price / (self.exports * self.home_price)
        home = TradingEconomyElement(
            'home',
            E=rate,
            X=self.exports,
            M=self.imports,
            PH=self.home_price,
            home_price=self.home_price,
            inflation=self.inflation,
            world_price=self.world_price,
            exports=self.exports,
            imports=self.imports,
            export_price=self.home_price * rate,
            import_price=self.world_price / rate,
            export_elasticity=self.export_elasticity,
            import_elasticity=self.import_elasticity,
            floating=float(self.floating),
            import_share=self.imports / (self.demand + self.government + self.exports - self.imports),
            demand=self.demand,
            government=self.government,
        )
        self.system_ = System(self.label, elements=[home], interactions=self._get_adjustments())
