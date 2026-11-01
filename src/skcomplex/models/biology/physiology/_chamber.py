"""Elastic compartment of blood, a heart chamber or a vascular bed, whose pressure is elastance times volume."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ChamberElement(Element):
    """Elastic compartment of blood, a heart chamber or a vascular bed, whose pressure is elastance times volume.

    The pressure is `E(t) (V - V0)`. The elastance of a heart chamber rises from `E_min` to `E_max` and back as a half
    sine during the systole that starts every `period`, and the elastance of a vessel stays at `E_min` when `E_max`
    equals it. Volumes are in ml, pressures in mmHg, and times in s.

    Args:
        label:
            The label of the chamber.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        V:
            The initial volume.

        V0:
            The unstressed volume.

        E_min:
            The elastance in diastole, the inverse of the compliance.

        E_max:
            The largest elastance in systole.

        period:
            The duration of a heartbeat.

        systole:
            The duration of the contraction.

    Examples:
        >>> from skcomplex.models.biology.physiology import ChamberElement
        >>> ChamberElement('arteries', V=150.0, E_min=0.8, E_max=0.8).E_max
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        V0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        E_min: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        E_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.8,
        systole: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.V = V
        self.V0 = V0
        self.E_min = E_min
        self.E_max = E_max
        self.period = period
        self.systole = systole
