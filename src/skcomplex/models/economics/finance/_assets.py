"""Finance models: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import NumericDomain, parameter, role, state
from ....core.components import ElementComponent
from ....core.interactions import BaseInteraction
from ....core.laws import BaseLaw, NoiseLaw, RateLaw, Symbols
from ....core.system import System
from ....engine.runtime import register_model


class AssetComponent(ElementComponent):
    """Traded asset described by its price.

    Args:
        label:
            The label of the asset.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.finance import AssetComponent
        >>> AssetComponent('stock', price=100.0).price
        100.0
    """

    price: float = state(default=1.0, domain=NumericDomain.POSITIVE)


class GeometricBrownianMotionInteraction(BaseInteraction):
    """Geometric Brownian motion, `dS = mu * S * dt + sigma * S * dW`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the asset.

        mu:
            The drift.

        sigma:
            The volatility.

    Examples:
        >>> from skcomplex.models.economics.finance import GeometricBrownianMotionInteraction
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', sigma=0.3).sigma
        0.3
    """

    target: str = role()
    mu: float = parameter(default=0.05, domain=NumericDomain.REAL)
    sigma: float = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the drift `mu * S` and the noise amplitude `sigma * S`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.price, s.mu * s.target.price), NoiseLaw(s.target.price, s.sigma * s.target.price)]


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
        components=[AssetComponent('asset', price=price)],
        interactions=[GeometricBrownianMotionInteraction('gbm', target='asset', mu=mu, sigma=sigma)],
    )
