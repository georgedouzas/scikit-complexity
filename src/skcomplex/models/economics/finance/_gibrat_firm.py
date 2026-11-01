"""Firm whose size is multiplied every step by an independent normal growth factor, `x = lambda x`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._gibrat_firm_view import GibratFirmView


class GibratFirmElement(Element):
    """Firm whose size is multiplied every step by an independent normal growth factor, `x = lambda x`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial size.

        previous:
            The size before the last step.

        mu:
            The mean of the growth factor.

        sigma:
            The standard deviation of the growth factor.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.finance import GibratFirmElement
        >>> GroupElement('firms', element=GibratFirmElement, size=10, values={'mu': 1.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 100.0,
        previous: Annotated[float, StateSpec()] = 100.0,
        mu: Annotated[float, ParameterSpec()] = 1.02,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.previous = previous
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def _get_view(cls) -> type[GibratFirmView]:
        """Return the view of the agents of the element."""
        return GibratFirmView

    @staticmethod
    def get_growth(agent: GibratFirmView, model: ModelView) -> float:
        """Return the size multiplied by a growth factor drawn from a normal distribution.

        Args:
            agent:
                The firm.

            model:
                The model view of the growth.

        Returns:
            size:
                The size after the step.
        """
        return agent.x * float(model.rng.normal(agent.mu, agent.sigma))
