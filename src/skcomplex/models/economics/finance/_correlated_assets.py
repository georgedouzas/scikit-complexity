"""Finance models: asset prices driven by correlated Wiener processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.system import System
from ....engine.solvers import EulerMaruyamaSolver
from ....specs import WienerProcessSpec
from ..._catalog import register_model
from ._asset import AssetElement
from ._geometric_brownian_motion import GeometricBrownianMotionInteraction


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
        >>> correlated_geometric_brownian_motion(rho=0.5).interactions[1].process.covariance
        [[1.0, 0.5], [0.5, 1.0]]
    """
    labels = ('first', 'second')
    market = WienerProcessSpec('market', dimension=2, covariance=[[1.0, rho], [rho, 1.0]])
    return System(
        elements=[AssetElement(label, price=price) for label, price in zip(labels, prices, strict=True)],
        interactions=[
            GeometricBrownianMotionInteraction(
                f'{label}_motion',
                target=label,
                process=market,
                index=index,
                mu=drift,
                sigma=volatility,
            )
            for index, (label, drift, volatility) in enumerate(zip(labels, mu, sigma, strict=True))
        ],
    )
