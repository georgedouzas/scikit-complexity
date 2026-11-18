"""Geometric Brownian motion of the price of an asset.

It follows `dS = mu * S * dt + sigma * S * dW`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction, WienerProcessSpec
from .....core.components.laws import NoiseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.specs import Role


class GeometricBrownianMotionInteraction(Interaction):
    """Geometric Brownian motion of the price of an asset.

    It follows `dS = mu * S * dt + sigma * S * dW`. The Wiener process `dW` is independent by default, or process
    `index` of a `WienerProcessSpec` that correlates it with the motions of other assets.

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
        >>> from skcomplex.core.components.interactions import WienerProcessSpec
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', sigma=0.3).sigma
        0.3
        >>> market = WienerProcessSpec('market', dimension=2)
        >>> GeometricBrownianMotionInteraction('gbm', target='stock', process=market, index=1).index
        1
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mu: float = 0.05,
        sigma: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
        process: WienerProcessSpec | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.sigma = sigma
        self.process = process
        self.index = index

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift `mu * S` and the noise `sigma * S dW` of the price as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            price = context.target.price
            process = None
            if self.process is not None:
                process = context.process.dW if self.index is None else context.process.dW[self.index]
            self.laws_: list[BaseComponent] = [
                RateLaw(price, context.mu * price),
                NoiseLaw(price, context.sigma * price, process=process),
            ]
