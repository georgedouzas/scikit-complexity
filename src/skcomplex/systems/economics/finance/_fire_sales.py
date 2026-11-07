"""Finance systems: fire sales of banks with overlapping portfolios, whose sales depress the prices of common assets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import FeedbackInteraction
from ....core.components.laws import BaseLaw
from ....core.components.system import System
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common._balance_sheet import DEFAULTED, SOLVENT, BalanceSheetElement, BalanceSheetView

SELLING_RULES = ('threshold', 'leverage')


class MarketAssetView(AgentView):
    """Asset as the fire sales see it: its price and the share of it liquidated."""

    price: float
    liquidated: float


class MarketAssetElement(Element):
    """Asset held by banks, whose price falls with the share of it liquidated, `p = exp(-alpha x)`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        price:
            The price of the asset, 1 before any sale.

        liquidated:
            The share of the asset sold so far.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import MarketAssetElement
        >>> GroupElement('assets', element=MarketAssetElement, size=4).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        liquidated: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
        self.liquidated = liquidated

    @classmethod
    def _get_view(cls) -> type[MarketAssetView]:
        """Return the view of the agents of the element."""
        return MarketAssetView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the fire sales update every asset at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class FireSaleInteraction(FeedbackInteraction[SystemView]):
    """Banks that sell common assets on losses, and the market impact that lowers the prices of what they sell.

    Every step the banks value their holdings at the current prices. Under the rule `'threshold'` a bank whose losses
    exhaust its equity defaults and liquidates its whole portfolio, and otherwise stays passive. Under `'leverage'` a
    bank sells the same share of every holding to bring its assets back to `leverage` times its equity, and liquidates
    everything when its equity is gone. The proceeds become cash, and the price of every asset falls to
    `p0 exp(-alpha x)`, where `x` is the share of the asset liquidated so far and `p0` its price after the shock.

    Args:
        label:
            The label of the fire sales.

        rule:
            Unused, since the fire sales override `act`.

        phase:
            The phase the fire sales act in, or `None` for the default phase.

        banks:
            The path of the group of banks.

        assets:
            The path of the group of assets.

        holdings:
            The initial holdings, `holdings[i][a]` the units of asset `a` that bank `i` holds.

        liabilities:
            The liabilities of every bank.

        alpha:
            The market impact.

        leverage:
            The target ratio of assets to equity of the rule `'leverage'`.

        selling:
            The rule of the banks, `'threshold'` or `'leverage'`.

    Raises:
        ValidationError:
            The rule is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.economics.finance import FireSaleInteraction
        >>> FireSaleInteraction('sales', banks='banks', assets='assets').reports
        ('defaults', 'sold')
    """

    reports: ClassVar[tuple[str, ...]] = ('defaults', 'sold')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        banks: Annotated[str, RoleSpec()],
        assets: Annotated[str, RoleSpec()],
        holdings: tuple[tuple[float, ...], ...] = (),
        liabilities: tuple[float, ...] = (),
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        leverage: Annotated[float, ParameterSpec(domain=RealSet.open(1, oo))] = 20.0,
        selling: str = 'threshold',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.banks = banks
        self.assets = assets
        self.holdings = holdings
        self.liabilities = liabilities
        self.alpha = alpha
        self.leverage = leverage
        self.selling = selling

    def _check(self) -> None:
        """Raise an error for an unknown rule.

        Raises:
            ValidationError:
                The rule is unknown.
        """
        super()._check()
        if self.selling not in SELLING_RULES:
            error_msg = f'The rule of {self.label!r} is {self.selling!r}. Use one of {SELLING_RULES}.'
            raise ValidationError(error_msg)

    def act(self, system: SystemView) -> None:
        """Value the portfolios, sell, and move the prices by the market impact.

        Args:
            system:
                The system view of the fire sales.
        """
        banks = sorted(system.agents('banks', BalanceSheetView), key=lambda bank: bank.id)
        assets = sorted(system.agents('assets', MarketAssetView), key=lambda asset: asset.id)
        initial = np.asarray(self.holdings, dtype=float)
        holdings = np.asarray(system.storage.get('holdings', initial))
        prices = np.array([asset.price for asset in assets])
        base = np.asarray(system.storage.get('base', prices))
        liquidated = np.array([asset.liquidated for asset in assets])
        cash = np.array([bank.external for bank in banks])
        status = np.array([bank.status for bank in banks])
        values = holdings @ prices + cash
        equity = values - np.asarray(self.liabilities, dtype=float)
        failed = equity <= 0
        if self.selling == 'leverage':
            excess = values - system.parameter('leverage') * equity
            excess = np.where(excess > 1e-9 * values, excess, 0.0)
            risky = holdings @ prices
            share = np.divide(excess, risky, out=np.zeros_like(excess), where=risky > 0)
            share = np.where(failed, 1.0, np.minimum(share, 1.0))
        else:
            share = np.where(failed & (status == SOLVENT), 1.0, 0.0)
        sold = holdings * share[:, None]
        cash = cash + sold @ prices
        holdings = holdings - sold
        totals = initial.sum(axis=0)
        liquidated = np.minimum(
            1.0,
            liquidated + np.divide(sold.sum(axis=0), totals, out=np.zeros_like(totals), where=totals > 0),
        )
        prices = base * np.exp(-system.parameter('alpha') * liquidated)
        status = np.where(failed, DEFAULTED, status)
        equity = holdings @ prices + cash - np.asarray(self.liabilities, dtype=float)
        system.storage.update(holdings=holdings, base=base)
        for bank, value, money, code in zip(banks, equity, cash, status, strict=True):
            bank.equity, bank.external, bank.status = float(value), float(money), float(code)
        for asset, price, share_sold in zip(assets, prices, liquidated, strict=True):
            asset.price, asset.liquidated = float(price), float(share_sold)
        system.report('defaults', float(np.sum(status == DEFAULTED)))
        system.report('sold', float((sold @ prices).sum()))


@register_system(
    field='economics/finance',
    tags=('systemic risk', 'fire sales', 'overlapping portfolios', 'bipartite network', 'market impact', 'agents'),
    references=('Bardoscia et al. 2021', 'Caccioli et al. 2014'),
    defaults={'until': 30},
)
def overlapping_portfolio_fire_sales(
    n_banks: int = 100,
    n_assets: int = 100,
    *,
    diversification: int = 10,
    leverage: float = 20.0,
    alpha: float = 1.0,
    shocked: int = 0,
    shock: float = 0.2,
    selling: str = 'threshold',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return banks with overlapping portfolios whose fire sales of common assets spread their losses to one another.

    Every bank invests equal amounts in `diversification` assets drawn at random, a bipartite network of banks and
    assets, with assets `leverage` times its equity. A shock lowers the price of one asset, and banks react to their
    losses: under the linear threshold rule a bank stays passive until its losses exhaust its equity and then
    liquidates its whole portfolio, and under leverage targeting it sells a share of every holding to restore its
    leverage. Sales lower prices through the market impact `p = p0 exp(-alpha x)` of the share `x` of an asset
    liquidated, which hits every bank holding the asset, so losses travel between banks that share no contract, the
    indirect contagion of Bardoscia et al. (2021, Section 3.2) after Caccioli et al. (2014). Larger impact and leverage
    make the cascades larger, and leverage targeting sells more for small shocks.

    Args:
        n_banks:
            The number of banks.

        n_assets:
            The number of assets.

        diversification:
            The number of assets in every portfolio.

        leverage:
            The ratio of assets to equity of every bank, and the target of leverage targeting.

        alpha:
            The market impact.

        shocked:
            The index of the asset the shock hits.

        shock:
            The share of its price the shocked asset loses.

        selling:
            The rule of the banks, `'threshold'` or `'leverage'`.

        random_state:
            The seed or generator of the portfolios.

    Returns:
        system:
            The system, with the groups `banks`, of the equity `banks.equity` and the cash `banks.external`, and
            `assets`, of the prices `assets.price`, and the fire sales `sales`, whose reporters are the number of
            defaulted banks `sales.defaults` and the value sold in the step `sales.sold`.

    Examples:
        >>> from skcomplex.systems.economics.finance import overlapping_portfolio_fire_sales
        >>> [group.prepare().size_ for group in overlapping_portfolio_fire_sales(10, 5, diversification=2).elements]
        [10, 5]
    """
    rng = np.random.default_rng(random_state)
    holdings = np.zeros((n_banks, n_assets))
    for bank in range(n_banks):
        holdings[bank, rng.choice(n_assets, diversification, replace=False)] = 1.0 / diversification
    assets_value = holdings.sum(axis=1)
    liabilities = assets_value * (1 - 1 / leverage)
    prices = np.ones(n_assets)
    prices[shocked] = 1 - shock
    equity = holdings @ prices - liabilities
    banks = GroupElement('banks', element=BalanceSheetElement, size=n_banks, values={'equity': equity})
    assets = GroupElement('assets', element=MarketAssetElement, size=n_assets, values={'price': prices})
    sales = FireSaleInteraction(
        'sales',
        banks='banks',
        assets='assets',
        holdings=tuple(map(tuple, holdings.tolist())),
        liabilities=tuple(liabilities.tolist()),
        alpha=alpha,
        leverage=leverage,
        selling=selling,
    )
    return System(elements=[banks, assets], interactions=[sales])
