"""Finance systems: asset prices driven by correlated Wiener processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import WienerProcessSpec
from ..common._asset import AssetElement
from ..common._geometric_brownian_motion import GeometricBrownianMotionInteraction


class CorrelatedGeometricBrownianMotionSystem(BaseSystem):
    """Two assets whose prices follow geometric Brownian motions.

    Correlated Wiener processes drive the motions. A `WienerProcessSpec` of dimension 2 with the covariance
    `[[1, rho], [rho, 1]]` drives the two prices, so their log returns over a short time have the correlation `rho`. The
    system has the assets `first` and `second`, whose motions share the Wiener processes `market`.

    Args:
        prices:
            The initial prices of the two assets.

        mu:
            The drifts of the two assets.

        sigma:
            The volatilities of the two assets.

        rho:
            The correlation of the two Wiener processes, from -1 to 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import CorrelatedGeometricBrownianMotionSystem
        >>> CorrelatedGeometricBrownianMotionSystem(rho=0.5).prepare().system_.interactions[1].process.covariance
        [[1.0, 0.5], [0.5, 1.0]]
    """

    field = 'economics/finance'
    topics = ('asset', 'sde', 'correlated noise')
    references = ('Black and Scholes 1973', 'Merton 1973')
    until = 1.0
    dt = 0.01
    solver = EulerMaruyamaSolver(dt=1e-3)

    def __init__(
        self,
        prices: tuple[float, float] = (100.0, 100.0),
        *,
        mu: tuple[float, float] = (0.05, 0.08),
        sigma: tuple[float, float] = (0.2, 0.3),
        rho: float = 0.7,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.prices = prices
        self.mu = mu
        self.sigma = sigma
        self.rho = rho

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        labels = ('first', 'second')
        market = WienerProcessSpec('market', dimension=2, covariance=[[1.0, self.rho], [self.rho, 1.0]])
        return CompositeSystem(
            self.label,
            elements=[AssetElement(label, price=price) for label, price in zip(labels, self.prices, strict=True)],
            interactions=[
                GeometricBrownianMotionInteraction(
                    f'{label}_motion',
                    target=label,
                    process=market,
                    index=index,
                    mu=drift,
                    sigma=volatility,
                )
                for index, (label, drift, volatility) in enumerate(zip(labels, self.mu, self.sigma, strict=True))
            ],
        )
