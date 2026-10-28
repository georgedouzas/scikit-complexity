"""Finance models: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


class AssetElement(Element):
    """Traded asset described by its price.

    Args:
        label:
            The label of the asset.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.finance import AssetElement
        >>> AssetElement('stock', price=100.0).price
        100.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price


class GeometricBrownianMotionInteraction(Interaction):
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

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mu: Annotated[float, ParameterSpec()] = 0.05,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        elements=[AssetElement('asset', price=price)],
        interactions=[GeometricBrownianMotionInteraction('gbm', target='asset', mu=mu, sigma=sigma)],
    )
