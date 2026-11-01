"""Reactor of the scaled Oregonator, the scaled concentrations of bromous acid, bromide, and the oxidized catalyst."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class OregonatorElement(Element):
    """Reactor of the scaled Oregonator, the scaled concentrations of bromous acid, bromide, and the oxidized catalyst.

    The scaled concentrations follow `epsilon dx/dt = q y - x y + x (1 - x)`, `delta dy/dt = -q y - x y + f z`, and
    `dz/dt = x - z`. The reactions of `oregonator` change the concentrations, with the time scales `epsilon` and
    `delta` dividing the stoichiometric coefficients of `x` and `y`, and read the parameters from the reactor.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial scaled concentration of bromous acid.

        y:
            The initial scaled concentration of bromide.

        z:
            The initial scaled concentration of the oxidized catalyst.

        epsilon:
            The ratio of the time scales of bromous acid and the catalyst.

        delta:
            The ratio of the time scales of bromide and the catalyst.

        q:
            The ratio of rate constants that sets the lowest bromous acid concentration.

        f:
            The stoichiometric factor of bromide production from the catalyst.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import OregonatorElement
        >>> OregonatorElement('reactor').epsilon
        0.04
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.04,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0004,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0008,
        f: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.epsilon = epsilon
        self.delta = delta
        self.q = q
        self.f = f
