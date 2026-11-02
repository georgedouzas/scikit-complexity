"""Finance models: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.system import System
from ..._catalog import register_model
from ._asset import AssetElement
from ._geometric_brownian_motion import GeometricBrownianMotionInteraction


@register_model(
    field='economics/finance',
    tags=('asset', 'sde'),
    references=('Black and Scholes 1973',),
    defaults={'until': 1.0, 'dt': 0.01},
)
def geometric_brownian_motion(price: float = 100.0, *, mu: float = 0.05, sigma: float = 0.2) -> System:
    """Return an asset whose price follows geometric Brownian motion.

    Args:
        price:
            The initial price.

        mu:
            The drift.

        sigma:
            The volatility.

    Returns:
        system:
            The model, with the asset `asset`.

    Examples:
        >>> from skcomplex.models.economics.finance import geometric_brownian_motion
        >>> geometric_brownian_motion(sigma=0.3).interactions[0].sigma
        0.3
    """
    return System(
        elements=[AssetElement('asset', price=price)],
        interactions=[GeometricBrownianMotionInteraction('gbm', target='asset', mu=mu, sigma=sigma)],
    )
