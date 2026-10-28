"""Finance models: asset prices driven by correlated Wiener processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import EulerMaruyamaSolver
from ....specs import ParameterSpec, RoleSpec, WienerProcessSpec
from ._assets import AssetElement


class CorrelatedGeometricBrownianMotionInteraction(Interaction):
    """Geometric Brownian motion `dS = mu * S * dt + sigma * S * dW_k` driven by process `k` of a `WienerProcessSpec`.

    Args:
        label:
            The label of the interaction.

        index:
            The process of the `WienerProcessSpec`.

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
        >>> from skcomplex.specs import WienerProcessSpec
        >>> from skcomplex.models.economics.finance import CorrelatedGeometricBrownianMotionInteraction
        >>> market = WienerProcessSpec('market', dimension=2)
        >>> CorrelatedGeometricBrownianMotionInteraction('gbm', target='stock', market=market, index=1).index
        1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        market: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
        mu: Annotated[float, ParameterSpec()] = 0.05,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        index: int = 0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.market = market
        self.mu = mu
        self.sigma = sigma
        self.index = index

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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

    A `WienerProcessSpec` of dimension 2 with the covariance `[[1, rho], [rho, 1]]` drives the two prices, so their log
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
    market = WienerProcessSpec('market', dimension=2, covariance=[[1.0, rho], [rho, 1.0]])
    return System(
        elements=[AssetElement(label, price=price) for label, price in zip(labels, prices, strict=True)],
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
