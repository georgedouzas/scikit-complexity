"""Walker at the position `x` whose diffusion coefficient `K alpha (t + t0)^(alpha - 1)` changes in time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ScaledWalkerElement(Element):
    """Walker at the position `x` whose diffusion coefficient `K alpha (t + t0)^(alpha - 1)` changes in time.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the walker is drawn, or `None` for none.

        x:
            The initial position.

        K:
            The generalized diffusion coefficient.

        alpha:
            The anomalous diffusion exponent.

        t0:
            The time offset of the diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ScaledWalkerElement
        >>> ScaledWalkerElement('walker', alpha=1.5).alpha
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 0.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        t0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.K = K
        self.alpha = alpha
        self.t0 = t0
