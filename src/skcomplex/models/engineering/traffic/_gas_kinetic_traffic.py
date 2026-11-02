"""Cell of a road with the traffic density `rho` and the flow `Q` of the gas-kinetic-based traffic model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class GasKineticTrafficElement(Element):
    """Cell of a road with the traffic density `rho` and the flow `Q` of the gas-kinetic-based traffic model.

    The density and the flow are conserved by conservation interactions, and the flow relaxes to a dynamic equilibrium
    speed that brakes on the density and the speed read ahead at the interaction point. Lengths are in kilometres and
    times in hours, so `T` and `tau` are fractions of an hour.

    Args:
        label:
            The label of the cell or the road.

        visual:
            How the cell is drawn, or `None` for `default_visual_`.

        rho:
            The initial density in vehicles per kilometre.

        Q:
            The initial flow in vehicles per hour.

        V0:
            The desired speed.

        T:
            The safe time headway.

        tau:
            The relaxation time of the speed.

        rho_max:
            The maximum density.

        gamma:
            The anticipation factor, the interaction distance in units of the safe distance `1 / rho_max + T V`.

        A0:
            The variance prefactor of free traffic.

        dA:
            The increase of the variance prefactor from free to congested traffic, halved.

        rho_c:
            The density of the transition of the variance prefactor.

        d_rho:
            The width of the transition of the variance prefactor.

    Examples:
        >>> from skcomplex.models.engineering.traffic import GasKineticTrafficElement
        >>> GasKineticTrafficElement('cell').gamma
        1.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 20.0,
        Q: Annotated[float, StateSpec()] = 1800.0,
        V0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 110.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.8 / 3600,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 35 / 3600,
        rho_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 160.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        A0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.008,
        dA: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        rho_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.27 * 160,
        d_rho: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05 * 160,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.Q = Q
        self.V0 = V0
        self.T = T
        self.tau = tau
        self.rho_max = rho_max
        self.gamma = gamma
        self.A0 = A0
        self.dA = dA
        self.rho_c = rho_c
        self.d_rho = d_rho
