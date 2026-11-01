"""Cell of a road with the traffic density `rho` and the flow `q` of the Payne model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PayneElement(Element):
    """Cell of a road with the traffic density `rho` and the flow `q` of the Payne model.

    The density and the flow are conserved by conservation interactions, and the flow relaxes to the Greenshields flow
    by a relaxation interaction.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        q: Annotated[object, StateSpec()] = 0.21,
        vmax: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.q = q
        self.vmax = vmax
        self.rho_max = rho_max
        self.tau = tau
