"""Define Stuart-Landau oscillators, the normal form of a Hopf bifurcation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.laws import BaseLaw, RateLaw
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class StuartLandauElement(Element):
    """Stuart-Landau oscillator `z' = tau ((mu + i omega) z - kappa (1 + i beta) |z|^2 z)` of `z = x + i y`.

    It is the normal form of a supercritical Hopf bifurcation. For `mu > 0` the oscillator settles on the limit cycle
    of radius `sqrt(mu / kappa)`, which it turns at the frequency `tau (omega - beta mu)`, and for `mu < 0` it decays to
    the fixed point 0. The twist `beta` makes the frequency depend on the amplitude, and the time scale `tau` speeds
    up or slows down the whole flow.

    Args:
        label:
            The label of the oscillator or the group.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial real part of `z`.

        y:
            The initial imaginary part of `z`.

        mu:
            The growth rate of the fixed point 0, the bifurcation parameter.

        omega:
            The frequency of the linear rotation.

        beta:
            The twist, the shear of the frequency with the amplitude.

        kappa:
            The strength of the cubic saturation.

        tau:
            The time scale.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.physics.common.oscillators import StuartLandauElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = System(elements=[StuartLandauElement('oscillator', x=0.1, mu=4.0)])
        >>> trajectory = simulate(system, until=10.0, dt=0.5)
        >>> round(math.hypot(trajectory.values('oscillator.x')[-1], trajectory.values('oscillator.y')[-1]), 4)
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec()] = 1.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        beta: Annotated[float, ParameterSpec()] = 0.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.beta = beta
        self.kappa = kappa
        self.tau = tau

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the real and imaginary parts of `z`.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `x` and `y`.
        """
        s = namespace
        saturation = s.kappa * (s.x**2 + s.y**2)
        return [
            RateLaw(s.x, s.tau * (s.mu * s.x - s.omega * s.y - saturation * (s.x - s.beta * s.y))),
            RateLaw(s.y, s.tau * (s.mu * s.y + s.omega * s.x - saturation * (s.y + s.beta * s.x))),
        ]
