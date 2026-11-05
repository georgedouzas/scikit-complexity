"""Well-mixed SIR population at a node of a network, whose individuals hop along each edge at the rate `D`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SubpopulationElement(Element):
    """Well-mixed SIR population at a node of a network, whose individuals hop along each edge at the rate `D`.

    Inside the node, `beta S I / N` individuals are infected and `gamma I` recover per unit time, with
    `N = S + I + R`. Hopping along every edge adds `D` times the graph Laplacian to every compartment.

    Args:
        label:
            The label of the subpopulation or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        S:
            The initial number of susceptible individuals.

        I:
            The initial number of infected individuals.

        R:
            The initial number of recovered individuals.

        beta:
            The infection rate.

        gamma:
            The recovery rate.

        D:
            The rate at which an individual hops along one edge.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import SubpopulationElement
        >>> SubpopulationElement('city', S=1000.0).beta
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.I = I
        self.R = R
        self.beta = beta
        self.gamma = gamma
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infections, recoveries, and hops of the node.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        infections = s.beta * s.S * s.I / (s.S + s.I + s.R)
        recoveries = s.gamma * s.I
        return [
            RateLaw(s.S, -infections + s.D * laplacian(s.S)),
            RateLaw(s.I, infections - recoveries + s.D * laplacian(s.I)),
            RateLaw(s.R, recoveries + s.D * laplacian(s.R)),
        ]
