"""Tumour with its vasculature and effector cells under bevacizumab and atezolizumab, with resistance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AngiogenesisImmunotherapyTumourElement(Element):
    """Tumour with its vasculature and effector cells under bevacizumab and atezolizumab, with resistance.

    The tumour `A` grows by the Hahnfeldt law toward its vascular capacity `Vb` and effector cells `E` kill it. The
    capacity is stimulated and inhibited by the tumour and destroyed by the antiangiogenic effect `UAn`, and the
    immunotherapy effect `UIm` boosts the effector cells. Each drug follows a two-compartment system with central amount
    `c` and peripheral amount `q`, in mg per kg, infused for `width` days every `period` days. Its effect is
    `c^g / ((f U50)^g + c^g)`, where `f = 1 + delta_r R` and `R` integrates `max(0, u_theta - c)`, so the drug loses
    potency while its concentration stays low. Volumes are in mm³, and time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `angiogenesis_immunotherapy` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial tumour volume.

        Vb:
            The initial vascular capacity.

        E:
            The initial effector cells, dimensionless.

        c_an:
            The initial central amount of the antiangiogenic drug.

        q_an:
            The initial peripheral amount of the antiangiogenic drug.

        R_an:
            The initial resistance integral of the antiangiogenic drug.

        c_im:
            The initial central amount of the immunotherapeutic drug.

        q_im:
            The initial peripheral amount of the immunotherapeutic drug.

        R_im:
            The initial resistance integral of the immunotherapeutic drug.

        v1:
            The growth constant of the tumour.

        v2:
            The spontaneous loss of vasculature.

        a1:
            The kill rate of the tumour by effector cells.

        xi_s:
            The stimulation of the vasculature by the tumour.

        xi_i:
            The inhibition of the vasculature by the tumour.

        a2:
            The highest destruction rate of the vasculature by the antiangiogenic drug.

        r1:
            The proliferation rate of effector cells.

        c1:
            The inverse of the tumour volume above which the tumour suppresses the effector cells.

        d1:
            The death rate of effector cells.

        s_e:
            The influx of effector cells.

        a3:
            The highest boost of effector cells by the immunotherapeutic drug.

        alpha_an:
            The distribution rate of the antiangiogenic drug.

        beta_an:
            The elimination rate of the antiangiogenic drug.

        k12_an:
            The rate from the central to the peripheral compartment of the antiangiogenic drug.

        k21_an:
            The rate from the peripheral to the central compartment of the antiangiogenic drug.

        u50_an:
            The amount of antiangiogenic drug of half effect before resistance.

        alpha_im:
            The distribution rate of the immunotherapeutic drug.

        beta_im:
            The elimination rate of the immunotherapeutic drug.

        k12_im:
            The rate from the central to the peripheral compartment of the immunotherapeutic drug.

        k21_im:
            The rate from the peripheral to the central compartment of the immunotherapeutic drug.

        u50_im:
            The amount of immunotherapeutic drug of half effect before resistance.

        g:
            The steepness of the drug effects.

        delta_r:
            The capacity of the tumour to resist the drugs.

        u_theta:
            The amount below which resistance develops.

        tox_an:
            The amount of antiangiogenic drug at which its toxicity turns exponential.

        tox_im:
            The amount of immunotherapeutic drug at which its toxicity turns exponential.

        dose_an:
            The dose of the antiangiogenic drug, in mg per kg.

        dose_im:
            The dose of the immunotherapeutic drug, in mg per kg.

        period:
            The time between the starts of doses.

        width:
            The duration of an infusion.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.systems.biology.oncology import AngiogenesisImmunotherapyTumourElement
        >>> AngiogenesisImmunotherapyTumourElement('tumour').Vb
        625.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 5000.0,
        Vb: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 625.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c_an: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        q_an: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R_an: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        c_im: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        q_im: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R_im: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        v1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        v2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        xi_s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.7,
        xi_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00873,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.35,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3e-05,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0005,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.37451,
        s_e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        alpha_an: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4811,
        beta_an: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0348,
        k12_an: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.223,
        k21_an: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.215,
        u50_an: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 11.4274,
        alpha_im: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5827,
        beta_im: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0271,
        k12_im: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        k21_im: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2455,
        u50_im: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 7.1903,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        delta_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        u_theta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        tox_an: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        tox_im: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        dose_an: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        dose_im: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        period: float = 21.0,
        width: float = 1 / 24,
        doses: int = 6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.Vb = Vb
        self.E = E
        self.c_an = c_an
        self.q_an = q_an
        self.R_an = R_an
        self.c_im = c_im
        self.q_im = q_im
        self.R_im = R_im
        self.v1 = v1
        self.v2 = v2
        self.a1 = a1
        self.xi_s = xi_s
        self.xi_i = xi_i
        self.a2 = a2
        self.r1 = r1
        self.c1 = c1
        self.d1 = d1
        self.s_e = s_e
        self.a3 = a3
        self.alpha_an = alpha_an
        self.beta_an = beta_an
        self.k12_an = k12_an
        self.k21_an = k21_an
        self.u50_an = u50_an
        self.alpha_im = alpha_im
        self.beta_im = beta_im
        self.k12_im = k12_im
        self.k21_im = k21_im
        self.u50_im = u50_im
        self.g = g
        self.delta_r = delta_r
        self.u_theta = u_theta
        self.tox_an = tox_an
        self.tox_im = tox_im
        self.dose_an = dose_an
        self.dose_im = dose_im
        self.period = period
        self.width = width
        self.doses = doses
