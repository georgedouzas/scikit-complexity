"""Node of an activity-driven network, susceptible at `i = 0` or infected at `i = 1`, in discrete time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ActiveHostElement(Element):
    """Node of an activity-driven network, susceptible at `i = 0` or infected at `i = 1`, in discrete time.

    A `TransitionInteraction` recovers an infected node with probability `mu` in a step, and another infects a
    susceptible node with probability `lam` by every infected neighbour along the edges of the previous step. A
    `TopologyInteraction` joins the node to other nodes with the probability `a`, its activity.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        i:
            Whether the node is infected.

        a:
            The activity, the probability of firing at a step.

        lam:
            The probability of infection along one edge.

        mu:
            The probability of recovery at a step.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import ActiveHostElement
        >>> ActiveHostElement('node').mu
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        i: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.i = i
        self.a = a
        self.lam = lam
        self.mu = mu
