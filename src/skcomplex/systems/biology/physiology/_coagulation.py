"""Coagulation cascade: thrombin amplified by positive feedback, switched off by protein C, and fibrinolysis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CoagulationElement(Element):
    """Coagulation cascade: thrombin amplified by positive feedback, switched off by protein C, and fibrinolysis.

    An injury of height `injury` from `injury_time` for `duration` activates thrombin, which activates more of itself
    through the feedback of the cascade, `dT/dt = j(t) + ka T^2 / (k_half^2 + T^2) - (d_t + k_c C) T`. Thrombin
    activates the inhibitor protein C, `dC/dt = s_c T - d_c C`, and converts fibrinogen to fibrin,
    `dF/dt = s_f T - k_p P F`. Fibrin activates plasmin, `dP/dt = s_p F - d_p P`, which dissolves the fibrin.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `coagulation_positive_feedback` read its parameters.

    Args:
        label:
            The label of the blood.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial thrombin.

        C:
            The initial activated protein C.

        F:
            The initial fibrin.

        P:
            The initial plasmin.

        ka:
            The maximal rate of the positive feedback.

        k_half:
            The thrombin at which the feedback is half its maximum.

        d_t:
            The inactivation rate of thrombin by antithrombin.

        k_c:
            The inactivation of thrombin per unit of protein C.

        s_c:
            The activation of protein C per unit of thrombin.

        d_c:
            The inactivation rate of protein C.

        s_f:
            The production of fibrin per unit of thrombin.

        k_p:
            The lysis of fibrin per unit of plasmin.

        s_p:
            The activation of plasmin per unit of fibrin.

        d_p:
            The inactivation rate of plasmin.

        injury:
            The rate of thrombin activation by the injury.

        injury_time:
            The start of the injury.

        duration:
            The duration of the injury.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CoagulationElement
        >>> CoagulationElement('blood').injury_time
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        F: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ka: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        k_half: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        d_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        s_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        d_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        s_f: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        s_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        d_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        injury: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        injury_time: float = 1.0,
        duration: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.C = C
        self.F = F
        self.P = P
        self.ka = ka
        self.k_half = k_half
        self.d_t = d_t
        self.k_c = k_c
        self.s_c = s_c
        self.d_c = d_c
        self.s_f = s_f
        self.k_p = k_p
        self.s_p = s_p
        self.d_p = d_p
        self.injury = injury
        self.injury_time = injury_time
        self.duration = duration
