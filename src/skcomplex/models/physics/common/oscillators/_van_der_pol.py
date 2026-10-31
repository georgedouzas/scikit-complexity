"""Define Van der Pol oscillators, relaxation oscillators with nonlinear damping and an optional periodic forcing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from .....core.elements import Element
from .....core.laws import BaseLaw, RateLaw
from .....core.laws.utils import derivative
from .....specs import ParameterSpec, StateSpec, VisualSpec


class VanDerPolElement(Element):
    """Van der Pol oscillator `x'' = mu (1 - x^2) x' - x + A cos(omega t)`, with an optional periodic forcing.

    The damping is negative for small amplitudes and positive for large ones, so the unforced oscillator, `A = 0`,
    settles on a limit cycle of amplitude near 2. A forcing of amplitude `A` and frequency `omega` locks it or drives it
    quasiperiodically or chaotically.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        mu:
            The strength of the nonlinear damping.

        A:
            The amplitude of the forcing.

        omega:
            The frequency of the forcing.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.physics.common.oscillators import VanDerPolElement
        >>> system = System(elements=[VanDerPolElement('oscillator', x=0.1)])
        >>> trajectory = Simulation(system, dt=0.1).check().prepare().run(until=50.0).trajectory_
        >>> round(float(abs(trajectory.values('oscillator.x')[-300:]).max()), 1)
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 2.0,
        x_d1: float = 0.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        A: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.mu = mu
        self.A = A
        self.omega = omega

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the acceleration of the oscillator.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the second derivative of `x`.
        """
        x = namespace.x
        damping = namespace.mu * (1 - x**2) * derivative(x, 1)
        return [RateLaw(x, damping - x + namespace.A * cos(namespace.omega * namespace.t))]
