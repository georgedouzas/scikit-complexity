"""Walker on the real line whose velocity is an autoregressive process of memory `alpha` and noise `sigma`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MemoryWalkerElement(Element):
    """Walker on the real line whose velocity is an autoregressive process of memory `alpha` and noise `sigma`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        v:
            The initial velocity.

        alpha:
            The memory of the velocity.

        sigma:
            The deviation of the noise of the velocity.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MemoryWalkerElement
        >>> MemoryWalkerElement('walker', alpha=0.3).alpha
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 0.0,
        v: Annotated[float, StateSpec(role='velocity')] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.alpha = alpha
        self.sigma = sigma
