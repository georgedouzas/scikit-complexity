"""Geometric Brownian motion, `dS = mu * S * dt + sigma * S * dW`, of the price of an asset."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.specs import ParameterSpec, RoleSpec, WienerProcessSpec


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
        >>> from skcomplex.systems.economics.finance import GeometricBrownianMotionInteraction
        >>> from skcomplex.core.specs import WienerProcessSpec
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', sigma=0.3).sigma
        0.3
        >>> market = WienerProcessSpec('market', dimension=2)
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', process=market, index=1).index
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
