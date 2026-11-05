"""Cell of a road with the traffic density `rho`, the free speed `vmax`, and the jam density `rho_max`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LighthillWhithamRichardsElement(Element):
    """Cell of a road with the traffic density `rho`, the free speed `vmax`, and the jam density `rho_max`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        vmax: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.vmax = vmax
        self.rho_max = rho_max
