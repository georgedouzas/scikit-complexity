"""Agent of the stochastic Schelling model, whose utility `a * rho + b` depends on the density of its block."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._dweller_view import DwellerView


class DwellerElement(Element):
    """Agent of the stochastic Schelling model, whose utility `a * rho + b` depends on the density of its block.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        block:
            The index of the agent's block, which `stochastic_schelling` sets after every move.

        beta:
            The inverse temperature of the moves.

        altruism:
            The weight of the change of the global utility against the agent's own utility.

        slope:
            The slope `a` of the utility in the block density.

        intercept:
            The intercept `b` of the utility.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.segregation import DwellerElement
        >>> group = GroupElement('dwellers', element=DwellerElement, size=10, values={'beta': 3.0})
        >>> group.label, group.element.__name__, group.size
        ('dwellers', 'DwellerElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        block: Annotated[float, StateSpec(domain=NN)] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        altruism: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        slope: Annotated[float, ParameterSpec()] = -4.0 / 3.0,
        intercept: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.block = block
        self.beta = beta
        self.altruism = altruism
        self.slope = slope
        self.intercept = intercept

    @classmethod
    def _get_view(cls) -> type[DwellerView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return DwellerView
