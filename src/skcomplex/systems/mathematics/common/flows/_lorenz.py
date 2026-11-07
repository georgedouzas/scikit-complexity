"""Define the Lorenz flow, the three-mode truncation of Rayleigh-Bénard convection."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from .....core.components.elements import Element
from .....core.components.laws import BaseLaw, RateLaw
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class LorenzElement(Element):
    """Lorenz flow `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`.

    A subclass that declares the states with a fractional `StateSpec(order=q)` gives the fractional Lorenz flow with
    Caputo derivatives of order `q`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.mathematics.common.flows import LorenzElement
        >>> system = System(elements=[LorenzElement('lorenz', x=0.0, y=0.0, z=2.0, beta=0.5)])
        >>> trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
        >>> [round(float(value), 4) for value in trajectory.values('lorenz.z')]
        [2.0, 1.5576, 1.2131]
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 1.0,
        z: Annotated[float, StateSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 10.0,
        rho: Annotated[float, ParameterSpec()] = 28.0,
        beta: Annotated[float, ParameterSpec()] = 8 / 3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three modes.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `x`, `y`, and `z`.
        """
        return [
            RateLaw(s.x, s.sigma * (s.y - s.x)),
            RateLaw(s.y, s.x * (s.rho - s.z) - s.y),
            RateLaw(s.z, s.x * s.y - s.beta * s.z),
        ]
