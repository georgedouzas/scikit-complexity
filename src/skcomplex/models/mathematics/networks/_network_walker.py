"""Walker on the nodes of a network, which counts the steps it takes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._network_walker_view import NetworkWalkerView


class NetworkWalkerElement(Element):
    """Walker on the nodes of a network, which counts the steps it takes.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        steps:
            The initial number of steps taken.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.mathematics.networks import NetworkWalkerElement
        >>> group = GroupElement('walkers', element=NetworkWalkerElement, size=5)
        >>> group.label, group.element.__name__, group.size
        ('walkers', 'NetworkWalkerElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'steps'}),
        *,
        steps: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.steps = steps

    @classmethod
    def _get_view(cls) -> type[NetworkWalkerView]:
        """Return the view of the agents, which types their states."""
        return NetworkWalkerView
