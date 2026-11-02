"""Firm whose size follows Kesten's process, `x = lambda x + b`, with uniform random `lambda` and `b`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._kesten_firm_view import KestenFirmView


class KestenFirmElement(Element):
    """Firm whose size follows Kesten's process, `x = lambda x + b`, with uniform random `lambda` and `b`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial size.

        lambda_low:
            The lower bound of the multiplicative factor.

        lambda_high:
            The upper bound of the multiplicative factor.

        b_low:
            The lower bound of the additive term.

        b_high:
            The upper bound of the additive term.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.finance import KestenFirmElement
        >>> GroupElement('firms', element=KestenFirmElement, size=10, values={'b_high': 2.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        lambda_low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.48,
        lambda_high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.48,
        b_low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b_high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.b_low = b_low
        self.b_high = b_high

    @classmethod
    def _get_view(cls) -> type[KestenFirmView]:
        """Return the view of the agents of the element."""
        return KestenFirmView

    @staticmethod
    def get_growth(agent: KestenFirmView, model: ModelView) -> float:
        """Return the size multiplied by a random factor plus a random term.

        Args:
            agent:
                The firm.

            model:
                The model view of the growth.

        Returns:
            size:
                The size after the step.
        """
        factor = float(model.rng.uniform(agent.lambda_low, agent.lambda_high))
        return factor * agent.x + float(model.rng.uniform(agent.b_low, agent.b_high))
