"""Normal and malignant cells in the bone marrow and the peripheral blood, and cancer stem cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class BoneMarrowBloodElement(Element):
    """Normal and malignant cells in the bone marrow and the peripheral blood, and cancer stem cells.

    Normal marrow cells renew as `r1 NBM log((b1 + b2) / (theta1 + NBM))`, mutate into blasts at the rate `m1` and
    into cancer stem cells at the rate `m3`, and move to the blood, where they arrive after the maturation time
    `tau1` with the survival `exp(-alpha1 tau1 - r2 APB(t - tau1))`. Blasts grow as `r3 ABM log(b3 / (theta3 + ABM))`
    and move to the blood at the rate `m4`. Cancer stem cells renew logistically and convert into blasts after the
    time `tau2`. Malignant cells inhibit normal ones in both places.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `bone_marrow_blood_leukaemia` read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        NBM:
            The initial normal marrow cells.

        NPB:
            The initial normal blood cells.

        ABM:
            The initial marrow blasts.

        APB:
            The initial blood blasts.

        AS:
            The initial cancer stem cells.

        r1:
            The growth rate of normal marrow cells.

        b1:
            The carrying capacity of normal marrow cells.

        b2:
            The carrying capacity of normal blood cells.

        theta1:
            The threshold of the renewal of normal marrow cells.

        m1:
            The mutation rate of normal marrow cells into blasts.

        m2:
            The largest release rate of normal marrow cells into the blood.

        theta2:
            The number of normal blood cells that halves the release.

        r2:
            The inhibition of the release by blood blasts.

        m3:
            The mutation rate of normal marrow cells into cancer stem cells.

        c1:
            The inhibition of normal marrow cells by marrow blasts.

        alpha1:
            The loss rate of normal cells during maturation.

        tau1:
            The maturation time of normal cells.

        c2:
            The inhibition of normal blood cells by blood blasts.

        d1:
            The death rate of normal blood cells.

        r3:
            The growth rate of marrow blasts.

        b3:
            The carrying capacity of marrow blasts.

        theta3:
            The threshold of the growth of marrow blasts.

        m4:
            The rate at which marrow blasts move to the blood.

        m5:
            The conversion rate of cancer stem cells into blasts.

        alpha2:
            The loss rate of converting stem cells.

        tau2:
            The conversion time of cancer stem cells into blasts.

        r4:
            The inhibition of the conversion by blood blasts.

        d2:
            The loss rate of blood blasts.

        r5:
            The renewal rate of cancer stem cells.

        b4:
            The reciprocal carrying capacity of cancer stem cells.

    Examples:
        >>> from skcomplex.models.biology.oncology import BoneMarrowBloodElement
        >>> BoneMarrowBloodElement('patient').tau1
        96.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        NBM: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10000000000000.0,
        NPB: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3500000000.0,
        ABM: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 400000000000.0,
        APB: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5000000000.0,
        AS: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100000.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.149,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 58000000000000.0,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000000000000.0,
        theta1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000000000.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0013,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 27000.0,
        theta2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000000000.0,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-12,
        m3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-07,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-14,
        alpha1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        tau1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 96.0,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-14,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0083,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00396,
        b3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 368800000000000.0,
        theta3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000000000.0,
        m4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0065,
        m5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        alpha2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        tau2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 576.0,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-12,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        r5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        b4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-06,
    ) -> None:
        super().__init__(label, visual=visual)
        self.NBM = NBM
        self.NPB = NPB
        self.ABM = ABM
        self.APB = APB
        self.AS = AS
        self.r1 = r1
        self.b1 = b1
        self.b2 = b2
        self.theta1 = theta1
        self.m1 = m1
        self.m2 = m2
        self.theta2 = theta2
        self.r2 = r2
        self.m3 = m3
        self.c1 = c1
        self.alpha1 = alpha1
        self.tau1 = tau1
        self.c2 = c2
        self.d1 = d1
        self.r3 = r3
        self.b3 = b3
        self.theta3 = theta3
        self.m4 = m4
        self.m5 = m5
        self.alpha2 = alpha2
        self.tau2 = tau2
        self.r4 = r4
        self.d2 = d2
        self.r5 = r5
        self.b4 = b4
