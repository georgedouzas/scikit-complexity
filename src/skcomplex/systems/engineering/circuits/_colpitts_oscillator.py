"""Colpitts oscillator in dimensionless form, whose transistor conducts below a threshold of its third variable."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ColpittsOscillatorElement(Element):
    """Colpitts oscillator in dimensionless form, whose transistor conducts below a threshold of its third variable.

    The oscillator follows `x' = omega y - a F(z)`, `y' = c - omega x - z - b y`, and `z' = epsilon (y - d)`, with the
    transistor characteristic `F(z) = max(0, -1 - z)`, which is chaotic for `a = 30`, `b = 1`, `c = 20`, `d = 1`, and
    `epsilon = 1`.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial first variable.

        y:
            The initial second variable.

        z:
            The initial third variable.

        omega:
            The natural frequency.

        a:
            The gain of the transistor.

        b:
            The damping of the second variable.

        c:
            The bias of the second variable.

        d:
            The offset of the third variable.

        epsilon:
            The time scale of the third variable.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import ColpittsOscillatorElement
        >>> ColpittsOscillatorElement('oscillator').a
        30.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.0,
        z: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec()] = 20.0,
        d: Annotated[float, ParameterSpec()] = 1.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three variables of the oscillator.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        conduction = where(s.z < -1, -1 - s.z, 0)
        return [
            RateLaw(s.x, s.omega * s.y - s.a * conduction),
            RateLaw(s.y, s.c - s.omega * s.x - s.z - s.b * s.y),
            RateLaw(s.z, s.epsilon * (s.y - s.d)),
        ]
