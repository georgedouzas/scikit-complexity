"""Finance models: asset prices driven by correlated Wiener processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, role
from ....core.components import Wiener
from ....core.interactions import Interaction
from ....core.laws import Law, Noise, Rate, Symbols
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ....core.system import System
from ._assets import Asset


class CorrelatedGeometricBrownianMotion(Interaction):
    """Geometric Brownian motion `dS = mu * S * dt + sigma * S * dW_k` driven by process `k` of a Wiener element.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the asset.

        market:
            The path of the Wiener element whose process drives the price.

        index:
            The process of the Wiener element.

        mu:
            The drift.

        sigma:
            The volatility.

    Examples:
        >>> from skcomplex.models.economics.finance import CorrelatedGeometricBrownianMotion
        >>> CorrelatedGeometricBrownianMotion('gbm', target='stock', market='market', index=1).index
        1
    """

    target: str = role()
    market: str = role()
    index: int = 0
    mu: float = parameter(default=0.05, domain=Domain.REAL)
    sigma: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift `mu * S` and the noise amplitude `sigma * S` on process `index` of the market.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        price = s.target.price
        return [Rate(price, s.mu * price), Noise(price, s.sigma * price, process=s.market.dW[self.index])]


@register_model(
    field='economics/finance',
    tags=('asset', 'sde', 'correlated noise'),
    references=('Black and Scholes 1973', 'Merton 1973'),
    defaults={'until': 1.0, 'dt': 0.01, 'solver': EulerMaruyama(dt=1e-3)},
)
def correlated_geometric_brownian_motion(
    prices: tuple[float, float] = (100.0, 100.0),
    *,
    mu: tuple[float, float] = (0.05, 0.08),
    sigma: tuple[float, float] = (0.2, 0.3),
    rho: float = 0.7,
) -> System:
    """Return two assets whose prices follow geometric Brownian motions driven by correlated Wiener processes.

    A Wiener element of dimension 2 with the covariance `[[1, rho], [rho, 1]]` drives the two prices, so their log
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
            The model, with the assets `first` and `second` and the Wiener element `market`.

    Examples:
        >>> from skcomplex.models.economics.finance import correlated_geometric_brownian_motion
        >>> correlated_geometric_brownian_motion(rho=0.5).elements[2].covariance
        [[1.0, 0.5], [0.5, 1.0]]
    """
    labels = ('first', 'second')
    return System(
        components=[
            *(Asset(label, price=price) for label, price in zip(labels, prices, strict=True)),
            Wiener('market', dimension=2, covariance=[[1.0, rho], [rho, 1.0]]),
        ],
        interactions=[
            CorrelatedGeometricBrownianMotion(
                f'{label}_motion',
                target=label,
                market='market',
                index=index,
                mu=drift,
                sigma=volatility,
            )
            for index, (label, drift, volatility) in enumerate(zip(labels, mu, sigma, strict=True))
        ],
    )
