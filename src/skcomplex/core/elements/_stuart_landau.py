"""Define Stuart-Landau oscillators, the normal form of a Hopf bifurcation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from ...specs import ParameterSpec, StateSpec, VisualSpec
from ..base import BasePart
from ..laws import RateLaw
from ._element import Element


class StuartLandauElement(Element):
    """Stuart-Landau oscillator, the normal form `z' = (mu + i omega) z - (a + i beta) |z|^2 z` of a Hopf bifurcation.

    The complex amplitude is `z = x + i y`. For `mu / a > 0` the oscillator settles on a limit cycle of radius
    `sqrt(mu / a)`, which it runs at the angular frequency `omega - beta mu / a`. For `mu < 0` and `a > 0` the origin is
    stable, and for `a < 0` the bifurcation is subcritical.

    Args:
        label:
            The label of the oscillator, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial real part of the amplitude.

        y:
            The initial imaginary part of the amplitude.

        mu:
            The linear growth rate, the distance from the bifurcation.

        omega:
            The linear angular frequency.

        a:
            The cubic saturation of the amplitude.

        beta:
            The nonlinear shift of the frequency with the amplitude.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.core.elements import StuartLandauElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> system = System(elements=[StuartLandauElement('z', x=0.1, mu=4.0)])
        >>> trajectory = Simulation(system, dt=0.1).check().prepare().run(until=10).trajectory_
        >>> x, y = trajectory.values('z.x')[-1], trajectory.values('z.y')[-1]
        >>> round(float((x**2 + y**2) ** 0.5), 3)
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec()] = 1.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 1.0,
        beta: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.a = a
        self.beta = beta

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the rates of the real and the imaginary part of the amplitude.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of `x` and `y`.
        """
        s = namespace
        square = s.x**2 + s.y**2
        return [
            RateLaw(s.x, s.mu * s.x - s.omega * s.y - square * (s.a * s.x - s.beta * s.y)),
            RateLaw(s.y, s.mu * s.y + s.omega * s.x - square * (s.a * s.y + s.beta * s.x)),
        ]
