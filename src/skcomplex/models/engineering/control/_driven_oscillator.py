"""Damped harmonic oscillator that a bounded force drives, `x'' = -omega^2 x - 2 zeta omega x' + force`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._cart_pendulum import FORCE_LIMIT


class DrivenOscillatorElement(Element):
    """Damped harmonic oscillator that a bounded force drives, `x'' = -omega^2 x - 2 zeta omega x' + force`.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        force:
            The force per unit mass, the control input.

        omega:
            The natural frequency.

        zeta:
            The damping ratio.

    Examples:
        >>> from skcomplex.models.engineering.control import DrivenOscillatorElement
        >>> DrivenOscillatorElement('oscillator', x=1.0).omega
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-FORCE_LIMIT, FORCE_LIMIT))] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        zeta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.force = force
        self.omega = omega
        self.zeta = zeta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, -(s.omega**2) * s.x - 2 * s.zeta * s.omega * derivative(s.x, 1) + s.force)]
