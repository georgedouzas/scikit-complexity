"""Node of a network whose state is the probability that it is infected, in quenched mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import neighbors
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CarrierElement(Element):
    """Node of a network whose state is the probability that it is infected, in quenched mean field.

    The node recovers at rate 1 and every infected neighbour infects it at rate `lam`, with the probabilities of the
    neighbours taken as independent, so `dp_i / dt = -p_i + lam (1 - p_i) sum_j A_ij p_j` over the neighbours `j`.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p:
            The initial probability of being infected.

        lam:
            The spreading rate, the ratio of the infection rate per edge and the recovery rate.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import CarrierElement
        >>> CarrierElement('node').lam
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the recovery and the infection by the neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.p, -s.p + s.lam * (1 - s.p) * neighbors(s.other.p))]
