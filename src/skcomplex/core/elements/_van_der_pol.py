"""Define Van der Pol oscillators, relaxation oscillators with nonlinear damping."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from ...specs import ParameterSpec, StateSpec, VisualSpec
from ..base import BasePart
from ..laws import RateLaw
from ..laws.utils import derivative
from ._element import Element


class VanDerPolElement(Element):
    """Van der Pol oscillator `x'' = mu (1 - x^2) x' - x + A cos(omega t)`, with an optional periodic forcing.

    Without forcing, the oscillator settles on a limit cycle of amplitude close to 2. For small `mu` the cycle is nearly
    harmonic with period `2 pi`, and for large `mu` it is a relaxation oscillation with period close to
    `(3 - 2 ln 2) mu`. The state `x` is of order 2, and its velocity is the companion state `x_d1`.

    Args:
        label:
            The label of the oscillator, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        mu:
            The strength of the nonlinear damping.

        A:
            The amplitude of the forcing.

        omega:
            The angular frequency of the forcing.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.core.elements import VanDerPolElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> system = System(elements=[VanDerPolElement('oscillator', x=0.5)])
        >>> trajectory = Simulation(system, dt=0.05).check().prepare().run(until=40).trajectory_
        >>> round(float(abs(trajectory.values('oscillator.x')[-200:]).max()), 1)
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

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the acceleration of the oscillator.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the second derivative of `x`.
        """
        s = namespace
        return [RateLaw(s.x, s.mu * (1 - s.x**2) * derivative(s.x, 1) - s.x + s.A * cos(s.omega * s.t))]
