"""Finance systems: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.system import BaseSystem, System
from ..common._asset import AssetElement
from ..common._geometric_brownian_motion import GeometricBrownianMotionInteraction


class GeometricBrownianMotionSystem(BaseSystem):
    """An asset whose price follows geometric Brownian motion.

    The system has the asset `asset`.

    Args:
        price:
            The initial price.

        mu:
            The drift.

        sigma:
            The volatility.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
        >>> GeometricBrownianMotionSystem(sigma=0.3).prepare().system_.interactions[0].sigma
        0.3
    """

    field = 'economics/finance'
    topics = ('asset', 'sde')
    references = ('Black and Scholes 1973',)
    until = 1.0
    dt = 0.01

    def __init__(
        self,
        price: float = 100.0,
        *,
        mu: float = 0.05,
        sigma: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.price = price
        self.mu = mu
        self.sigma = sigma

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        return System(
            self.label,
            elements=[AssetElement('asset', price=self.price)],
            interactions=[GeometricBrownianMotionInteraction('gbm', target='asset', mu=self.mu, sigma=self.sigma)],
        )
