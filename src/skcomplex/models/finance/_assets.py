"""Model asset prices as stochastic processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core import Domain, parameter, role, state
from ...model import Element, Interaction, Law, Noise, Rate, Symbols


class Asset(Element):
    """Traded asset described by its price.

    Args:
        label:
            The label of the asset.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.finance import Asset
        >>> Asset('stock', price=100.0).price
        100.0
    """

    price: float = state(1.0, domain=Domain.POSITIVE)


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
        >>> from skcomplex.models.finance import GeometricBrownianMotion
        >>> GeometricBrownianMotion('gbm', target='stock', sigma=0.3).sigma
        0.3
    """

    target: str = role()
    mu: float = parameter(0.05, domain=Domain.REAL)
    sigma: float = parameter(0.2, domain=Domain.NONNEGATIVE)

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
