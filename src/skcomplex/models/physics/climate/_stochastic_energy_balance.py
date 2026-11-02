"""Global mean surface temperature of the Sutera-Fraedrich energy balance, with drifting and ramped controls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec

C0 = 1.10869e-8


C1 = 9.7137e9


class StochasticEnergyBalanceElement(Element):
    """Global mean surface temperature of the Sutera-Fraedrich energy balance, with drifting and ramped controls.

    The temperature follows `dT = c0 [-T^4 + c1 mu (b2 T^2 - a2)] dt + sqrt(nu) dW`, in kelvin and years. The control
    `mu` drifts at the rate `-drift`, and `b2 = b_ini + lam (b_fin - b_ini)` follows the logistic ramp
    `dlam/dt = rho lam (1 - lam)`. The control `a2 = a_ini + c1 mu (b2^2 - b_ini^2) / 4` moves with `b2` so that the
    discriminant `(c1 mu b2)^2 - 4 c1 mu a2` of the equilibria stays constant during the ramp.

    Args:
        label:
            The label of the climate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial temperature in kelvin.

        mu:
            The initial relative solar input.

        lam:
            The initial progress of the ramp, between 0 and 1.

        drift:
            The rate at which `mu` decreases, per year.

        rho:
            The rate of the logistic ramp of `b2`, per year.

        b_ini:
            The initial value of `b2`, per square kelvin.

        b_fin:
            The final value of `b2`, per square kelvin.

        a_ini:
            The initial value of `a2`.

        nu:
            The variance of the noise per year, in square kelvin.

        c0:
            The scale of the energy balance, per cubic kelvin and year.

        c1:
            The scale of the absorbed radiation, in kelvin to the fourth.

    Examples:
        >>> from skcomplex.models.physics.climate import StochasticEnergyBalanceElement
        >>> StochasticEnergyBalanceElement('earth').nu
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 290.0,
        mu: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        lam: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        drift: Annotated[float, ParameterSpec()] = 0.0,
        rho: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b_ini: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.69e-05,
        b_fin: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.69e-05,
        a_ini: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6927,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = C0,
        c1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = C1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.mu = mu
        self.lam = lam
        self.drift = drift
        self.rho = rho
        self.b_ini = b_ini
        self.b_fin = b_fin
        self.a_ini = a_ini
        self.nu = nu
        self.c0 = c0
        self.c1 = c1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the energy balance with its noise, the drift of `mu`, and the logistic ramp of `b2` and `a2`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        b2 = s.b_ini + s.lam * (s.b_fin - s.b_ini)
        a2 = s.a_ini + s.c1 * s.mu * (b2**2 - s.b_ini**2) / 4
        return [
            RateLaw(s.T, s.c0 * (-(s.T**4) + s.c1 * s.mu * (b2 * s.T**2 - a2))),
            NoiseLaw(s.T, s.nu**0.5),
            RateLaw(s.mu, -s.drift),
            RateLaw(s.lam, s.rho * s.lam * (1 - s.lam)),
        ]
