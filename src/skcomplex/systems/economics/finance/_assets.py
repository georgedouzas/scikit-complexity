"""Finance systems: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.components import Observable
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ..common.assets import AssetElement, GeometricBrownianMotionInteraction


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.price = price
        self.mu = mu
        self.sigma = sigma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[AssetElement('asset', price=self.price)],
            interactions=[GeometricBrownianMotionInteraction('gbm', target='asset', mu=self.mu, sigma=self.sigma)],
        )
