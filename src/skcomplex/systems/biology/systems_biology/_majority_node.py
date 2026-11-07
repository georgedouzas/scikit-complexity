"""Node of a noisy Boolean network, a state of plus or minus one that follows the majority of its inputs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MajorityNodeElement(Element):
    """Node of a noisy Boolean network, a state of plus or minus one that follows the majority of its inputs.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sigma:
            The state, -1 or 1.

        eta:
            The probability that the node takes the state opposite to the majority.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import MajorityNodeElement
        >>> MajorityNodeElement('node', eta=0.1).eta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        sigma: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([-1, 1]))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sigma = sigma
        self.eta = eta
