"""Finance models: asset prices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import RateInteraction
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec, WienerProcessSpec
from ..._catalog import register_model


class AssetElement(Element):
    """Traded asset described by its price.

    Args:
        label:
            The label of the asset.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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


class GeometricBrownianMotionInteraction(RateInteraction):
    """Geometric Brownian motion, `dS = mu * S * dt + sigma * S * dW`, of the price of an asset.

    The Wiener process `dW` is independent by default, or process `index` of a `WienerProcessSpec` that correlates it
    with the motions of other assets.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the asset.

        mu:
            The drift.

        sigma:
            The volatility.

        process:
            The Wiener processes, one of which drives the price, or `None` for an independent process.

        index:
            The process of `process` that drives the price, or `None` without processes.

    Examples:
        >>> from skcomplex.models.economics.finance import GeometricBrownianMotionInteraction
        >>> from skcomplex.specs import WienerProcessSpec
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', sigma=0.3).sigma
        0.3
        >>> market = WienerProcessSpec('market', dimension=2)
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', process=market, index=1).check().index
        1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mu: Annotated[float, ParameterSpec()] = 0.05,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        process: Annotated[WienerProcessSpec | None, RoleSpec(noise=True)] = None,
        index: int | None = None,
    ) -> None:
        super().__init__(label, target=target, state='price', process=process, index=index)
        self.mu = mu
        self.sigma = sigma

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the drift `mu * S`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The drift.
        """
        return namespace.mu * namespace.target.price

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the noise amplitude `sigma * S`.

        Args:
            namespace:
                The namespace of the symbols the noise can use.

        Returns:
            noise:
                The noise amplitude.
        """
        return namespace.sigma * namespace.target.price


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
