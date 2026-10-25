"""Finance models: asset prices driven by correlated Wiener processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from ....core.base import BaseInteraction, BaseLaw, NumericDomain, parameter, role
from ....core.components import WienerProcess
from ....core.laws import NoiseLaw, RateLaw, Symbols
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import EulerMaruyamaSolver
from ._assets import AssetComponent


class CorrelatedGeometricBrownianMotionInteraction(BaseInteraction):
    """Geometric Brownian motion `dS = mu * S * dt + sigma * S * dW_k` driven by process `k` of a `WienerProcess`.

    Args:
        label:
            The label of the interaction.

        index:
            The process of the `WienerProcess`.

        **values:
            The values of the declared variables, by name:

            target:
                The path of the asset.

            market:
                The Wiener processes, one of which drives the price.

            mu:
                The drift.

            sigma:
                The volatility.

    Examples:
        >>> from skcomplex.core.components import WienerProcess
        >>> from skcomplex.models.economics.finance import CorrelatedGeometricBrownianMotionInteraction
        >>> market = WienerProcess('market', dimension=2)
        >>> CorrelatedGeometricBrownianMotionInteraction('gbm', target='stock', market=market, index=1).index
        1
    """

    target: str = role()
    market: WienerProcess = role(noise=True)
    mu: float = parameter(default=0.05, domain=NumericDomain.REAL)
    sigma: float = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)

    def __init__(
        self,
        label: str,
        *,
        index: int = 0,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.index = index

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the drift `mu * S` and the noise amplitude `sigma * S` on process `index` of the market.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        price = s.target.price
        return [RateLaw(price, s.mu * price), NoiseLaw(price, s.sigma * price, process=s.market.dW[self.index])]


@register_model(
    field='economics/finance',
    tags=('asset', 'sde', 'correlated noise'),
    references=('Black and Scholes 1973', 'Merton 1973'),
    defaults={'until': 1.0, 'dt': 0.01, 'solver': EulerMaruyamaSolver(dt=1e-3)},
)
def correlated_geometric_brownian_motion(
    prices: tuple[float, float] = (100.0, 100.0),
    *,
    mu: tuple[float, float] = (0.05, 0.08),
    sigma: tuple[float, float] = (0.2, 0.3),
    rho: float = 0.7,
) -> System:
    """Return two assets whose prices follow geometric Brownian motions driven by correlated Wiener processes.

    A `WienerProcess` of dimension 2 with the covariance `[[1, rho], [rho, 1]]` drives the two prices, so their log
    returns over a short time have the correlation `rho`.

    Args:
        prices:
            The initial prices of the two assets.

        mu:
            The drifts of the two assets.

        sigma:
            The volatilities of the two assets.

        rho:
            The correlation of the two Wiener processes, from -1 to 1.

    Returns:
        system:
            The model, with the assets `first` and `second`, whose motions share the Wiener processes `market`.

    Examples:
        >>> from skcomplex.models.economics.finance import correlated_geometric_brownian_motion
        >>> correlated_geometric_brownian_motion(rho=0.5).interactions[1].market.covariance
        [[1.0, 0.5], [0.5, 1.0]]
    """
    labels = ('first', 'second')
    market = WienerProcess('market', dimension=2, covariance=[[1.0, rho], [rho, 1.0]])
    return System(
        components=[AssetComponent(label, price=price) for label, price in zip(labels, prices, strict=True)],
        interactions=[
            CorrelatedGeometricBrownianMotionInteraction(
                f'{label}_motion',
                target=label,
                market=market,
                index=index,
                mu=drift,
                sigma=volatility,
            )
            for index, (label, drift, volatility) in enumerate(zip(labels, mu, sigma, strict=True))
        ],
    )
