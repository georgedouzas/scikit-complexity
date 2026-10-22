"""Finance models: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Law, Noise, Rate, Symbols, System
from ....core.runtime import register_model


class Asset(Element):
    """Traded asset described by its price.

    Args:
        label:
            The label of the asset.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.finance import Asset
        >>> Asset('stock', price=100.0).price
        100.0
    """

    price: float = state(default=1.0, domain=Domain.POSITIVE)


class GeometricBrownianMotion(Interaction):
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
        >>> from skcomplex.models.economics.finance import GeometricBrownianMotion
        >>> GeometricBrownianMotion('gbm', target='stock', sigma=0.3).sigma
        0.3
    """

    target: str = role()
    mu: float = parameter(default=0.05, domain=Domain.REAL)
    sigma: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift `mu * S` and the noise amplitude `sigma * S`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.price, s.mu * s.target.price), Noise(s.target.price, s.sigma * s.target.price)]


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
        components=[Asset('asset', price=price)],
        interactions=[GeometricBrownianMotion('gbm', target='asset', mu=mu, sigma=sigma)],
    )
