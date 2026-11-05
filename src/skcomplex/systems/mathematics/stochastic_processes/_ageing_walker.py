"""Walker that steps up with a probability that decays with the number of its earlier up steps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._ageing_walker_view import AgeingWalkerView


class AgeingWalkerElement(Element):
    """Walker that steps up with a probability that decays with the number of its earlier up steps.

    After `k` up steps, the next step is up with probability `p / (1 + k)^nu` and down otherwise. The walker remembers
    its number of up steps at the end of the last step, from which its step follows.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        ups:
            The initial number of up steps.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability with the number of up steps.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import AgeingWalkerElement
        >>> group = GroupElement('walkers', element=AgeingWalkerElement, size=10, values={'nu': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('walkers', 'AgeingWalkerElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'x'}),
        *,
        x: Annotated[float, StateSpec(domain=ZZ, role='position')] = 0.0,
        ups: Annotated[float, StateSpec(domain=NN, lags=1)] = 0.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.ups = ups
        self.p = p
        self.nu = nu

    @classmethod
    def _get_view(cls) -> type[AgeingWalkerView]:
        """Return the view of the agents, which types their states and parameters."""
        return AgeingWalkerView
