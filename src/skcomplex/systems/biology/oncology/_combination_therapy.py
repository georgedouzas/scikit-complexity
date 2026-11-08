"""Chemotherapy combined with other therapies.

The other therapies are immunotherapy, viruses, antibodies, and antiangiogenic drugs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, log, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction

INFUSIONS = ('constant', 'exponential', 'periodic')

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class AngiogenesisImmunotherapyTumourElement(Element):
    """Tumour, vasculature, and effector cells under bevacizumab and atezolizumab.

    The tumour develops resistance.

    The tumour `A` grows by the Hahnfeldt law toward its vascular capacity `Vb` and effector cells `E` kill it. The
    capacity is stimulated and inhibited by the tumour and destroyed by the antiangiogenic effect `UAn`, and the
    immunotherapy effect `UIm` boosts the effector cells. Each drug follows a two-compartment system with central amount
    `c` and peripheral amount `q`, in mg per kg, infused for `width` days every `period` days. Its effect is
    `c^g / ((f U50)^g + c^g)`, where `f = 1 + delta_r R` and `R` integrates `max(0, u_theta - c)`, so the drug loses
    potency while its concentration stays low. Volumes are in mm³, and time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `AngiogenesisImmunotherapySystem` read its parameters.

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


class ChemoimmunotherapyTumourElement(Element):
    """Tumour with NK cells, CD8 T cells, and circulating lymphocytes under therapy.

    The therapy is chemotherapy, IL-2, and TIL injection.

    The tumour `A` grows logistically and NK cells `N` and CD8 T cells `E` kill it. CD8 T cells kill at the rate
    `D = a1 (E / A)^l / (g1 + (E / A)^l)`. The circulating lymphocytes `C` feed the NK cells. The chemotherapy `M`
    kills a fraction `1 - exp(-M)` of every population at its own rate, IL-2 `I` stimulates the CD8 T cells, and a TIL
    injection adds CD8 T cells. All three drugs follow the same schedule of doses at the rates `u_til`, `u_il2`, and
    `u_chemo`. Cells are counted, and time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `ChemoimmunotherapySystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial number of tumour cells.

        N:
            The initial number of NK cells.

        E:
            The initial number of CD8 T cells.

        C:
            The initial number of circulating lymphocytes.

        I:
            The initial amount of IL-2.

        M:
            The initial amount of chemotherapeutic drug.

        r1:
            The growth rate of the tumour.

        b1:
            The reciprocal carrying capacity of the tumour.

        c1:
            The rate at which NK cells kill tumour cells.

        a1:
            The highest rate at which CD8 T cells kill tumour cells.

        g1:
            The half-saturation of the CD8 T cell kill.

        l:
            The exponent of the CD8 T cell kill.

        a2:
            The chemotherapy kill rate of tumour cells.

        m1:
            The rate at which circulating lymphocytes become NK cells.

        d1:
            The death rate of NK cells.

        r2:
            The highest recruitment rate of NK cells by the tumour.

        g2:
            The half-saturation of the NK cell recruitment, in cells².

        psi1:
            The rate at which the tumour inactivates NK cells.

        a3:
            The chemotherapy kill rate of NK cells.

        r3:
            The highest recruitment rate of CD8 T cells by the tumour kill.

        g3:
            The half-saturation of the CD8 T cell recruitment, in cells².

        d2:
            The death rate of CD8 T cells.

        psi2:
            The rate at which the tumour inactivates CD8 T cells.

        psi3:
            The rate at which NK cells regulate CD8 T cells.

        r4:
            The rate at which the NK cell kill of tumour cells activates CD8 T cells.

        r5:
            The rate at which circulating lymphocytes meeting tumour cells activate CD8 T cells.

        a4:
            The chemotherapy kill rate of CD8 T cells.

        r6:
            The highest rate at which IL-2 stimulates CD8 T cells.

        g4:
            The half-saturation of the IL-2 stimulation.

        s1:
            The influx of circulating lymphocytes.

        d3:
            The death rate of circulating lymphocytes.

        a5:
            The chemotherapy kill rate of circulating lymphocytes.

        d_u1:
            The decay rate of IL-2.

        d_u2:
            The decay rate of the chemotherapeutic drug.

        u_til:
            The rate of TIL injection during a dose, in cells per day.

        u_il2:
            The rate of IL-2 injection during a dose.

        u_chemo:
            The rate of chemotherapy infusion during a dose.

        start:
            The time of the first dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ChemoimmunotherapyTumourElement
        >>> ChemoimmunotherapyTumourElement('tumour').r1
        0.431
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000000.0,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50000.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 11000000.0,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        M: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.431,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.17e-08,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 7.13e-10,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.17,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.618,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.657,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00129,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0412,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4198,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20200000.0,
        psi1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-07,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.996,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 303000.0,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        psi2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.42e-10,
        psi3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.8e-08,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.1e-07,
        r5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3e-11,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        r6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.125,
        g4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20000000.0,
        s1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 121000.0,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.012,
        a5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        d_u1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        d_u2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.9,
        u_til: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10000000.0,
        u_il2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10000000.0,
        u_chemo: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        start: float = 7.0,
        period: float = 7.0,
        width: float = 1.0,
        doses: int = 3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.N = N
        self.E = E
        self.C = C
        self.I = I
        self.M = M
        self.r1 = r1
        self.b1 = b1
        self.c1 = c1
        self.a1 = a1
        self.g1 = g1
        self.l = l
        self.a2 = a2
        self.m1 = m1
        self.d1 = d1
        self.r2 = r2
        self.g2 = g2
        self.psi1 = psi1
        self.a3 = a3
        self.r3 = r3
        self.g3 = g3
        self.d2 = d2
        self.psi2 = psi2
        self.psi3 = psi3
        self.r4 = r4
        self.r5 = r5
        self.a4 = a4
        self.r6 = r6
        self.g4 = g4
        self.s1 = s1
        self.d3 = d3
        self.a5 = a5
        self.d_u1 = d_u1
        self.d_u2 = d_u2
        self.u_til = u_til
        self.u_il2 = u_il2
        self.u_chemo = u_chemo
        self.start = start
        self.period = period
        self.width = width
        self.doses = doses


class ChemovirotherapyTumourElement(Element):
    """Uninfected and infected tumour cells, an oncolytic virus, and a drug.

    The drug is chemotherapeutic.

    The uninfected cells `AU` grow logistically, the virus `V` infects them, and infected cells `AV` burst at the rate
    `a2`, releasing `beta_v` virions each. The drug `U` kills both kinds of cells and is infused at the rate `u1`,
    constantly, as `u1 exp(-d_u t)`, or as `u1 sin(d_u t)^2`. One unit of cells is a million cells, and one unit of time
    is two days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `ChemovirotherapySystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AU:
            The initial number of uninfected cells.

        AV:
            The initial number of infected cells.

        V:
            The initial number of virions.

        U:
            The initial amount of drug.

        r1:
            The growth rate of uninfected cells.

        b:
            The reciprocal carrying capacity.

        alpha:
            The infection rate.

        a1:
            The drug kill rate of uninfected cells.

        a2:
            The rate at which the virus kills infected cells.

        a3:
            The drug kill rate of infected cells.

        beta_v:
            The burst size.

        d1:
            The decay rate of the virus.

        d_u:
            The decay rate of the drug.

        u1:
            The infusion rate of the drug.

        infusion:
            The form of the infusion, `'constant'`, `'exponential'`, or `'periodic'`.

    Raises:
        ParameterError:
            The infusion is not one of the known forms.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ChemovirotherapyTumourElement
        >>> ChemovirotherapyTumourElement('tumour', infusion='periodic').infusion
        'periodic'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.206,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1e-06,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5115,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.006,
        beta_v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d_u: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.16,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        infusion: str = 'constant',
    ) -> None:
        super().__init__(label, visual=visual)
        self.AU = AU
        self.AV = AV
        self.V = V
        self.U = U
        self.r1 = r1
        self.b = b
        self.alpha = alpha
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.beta_v = beta_v
        self.d1 = d1
        self.d_u = d_u
        self.u1 = u1
        self.infusion = infusion

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown infusion.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.

        Raises:
            ParameterError:
                The infusion is not one of the known forms.
        """
        super()._prepare(context)
        if self.infusion not in INFUSIONS:
            error_msg = f'The infusion must be one of {INFUSIONS}, got {self.infusion!r}.'
            raise ParameterError(error_msg)


class ClonalTumourElement(Element):
    """Two competing tumour clones and an immune response under therapy.

    The therapy is chemotherapy and a dendritic-cell vaccine.

    Each clone grows logistically, competes with the other clone and with the immune cells `E`, and dies at the
    chemotherapy rate `u_chemo1` or `u_chemo2` and at the rate `b UIm E` of the vaccine-recruited immune cells. The
    immune cells grow logistically to `theta` and are recruited by both clones. Cells are counted, and time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `ClonalChemoimmunotherapySystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A1:
            The initial number of cells of the first clone.

        A2:
            The initial number of cells of the second clone.

        E:
            The initial number of immune cells.

        r1:
            The growth rate of the first clone.

        r2:
            The growth rate of the second clone.

        b1:
            The reciprocal carrying capacity of the first clone.

        b2:
            The reciprocal carrying capacity of the second clone.

        c1:
            The competition of the second clone on the first.

        c2:
            The competition of the immune cells on the first clone.

        c3:
            The competition of the first clone on the second.

        c4:
            The competition of the immune cells on the second clone.

        r3:
            The growth rate of the immune cells.

        theta:
            The immune threshold.

        s1:
            The rate at which the first clone recruits immune cells.

        s2:
            The rate at which the second clone recruits immune cells.

        u_chemo1:
            The chemotherapy kill rate of the first clone.

        u_chemo2:
            The chemotherapy kill rate of the second clone.

        u_vaccine:
            The level of the immunotherapeutic agent.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ClonalTumourElement
        >>> ClonalTumourElement('tumour').theta
        300000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 90000.0,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 40000.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 500.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.11,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2e-08,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2e-08,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.5,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.5,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        theta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 300000.0,
        s1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0018,
        s2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0018,
        u_chemo1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        u_chemo2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        u_vaccine: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A1 = A1
        self.A2 = A2
        self.E = E
        self.r1 = r1
        self.r2 = r2
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.r3 = r3
        self.theta = theta
        self.s1 = s1
        self.s2 = s2
        self.u_chemo1 = u_chemo1
        self.u_chemo2 = u_chemo2
        self.u_vaccine = u_vaccine


class GliomaBrainElement(Element):
    """Glia, neurons, and a glioma under temozolomide and an oncolytic virus.

    Normal glia `G` and the glioma compete. Neurons `N` die in proportion to the loss of glia,
    `psi dG/dt H(-dG/dt) N`. The drug `U` kills glia, neurons, and glioma with saturating kills. The virus `V` infects
    only glioma cells, and the infected cells `AV` burst. Cells and virions are counted, the drug is in mg per m², and
    time is in days.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `GliomaChemovirotherapySystem` read its parameters.

    Args:
        label:
            The label of the brain.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        G:
            The initial number of glial cells.

        N:
            The initial number of neurons.

        AU:
            The initial number of uninfected glioma cells.

        AV:
            The initial number of infected glioma cells.

        V:
            The initial number of virions.

        U:
            The initial amount of drug.

        r1:
            The growth rate of glia.

        r2:
            The growth rate of the glioma.

        b:
            The reciprocal carrying capacity of glia and glioma.

        c1:
            The competition of the glioma on glia.

        c2:
            The competition of glia on the glioma.

        a1:
            The drug kill rate of glia.

        a2:
            The drug kill rate of neurons.

        a3:
            The drug kill rate of the glioma.

        g1:
            The half-saturation of the drug kill of glia.

        g2:
            The half-saturation of the drug kill of neurons.

        g3:
            The half-saturation of the drug kill of the glioma.

        psi:
            The loss of neurons per glial cell lost.

        alpha:
            The infection rate.

        a4:
            The rate at which the virus kills infected cells.

        beta_v:
            The burst size.

        d_u1:
            The decay rate of the virus.

        d_u2:
            The clearance rate of the drug.

        u1:
            The infusion rate of the drug, in mg per m² per day.

    Examples:
        >>> from skcomplex.systems.biology.oncology import GliomaBrainElement
        >>> GliomaBrainElement('brain').u1
        400.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5940000000000.0,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5940000000000.0,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 60000000000.0,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000000000.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0068,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.012,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.67e-13,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0769e-15,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0769e-16,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280800.0,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280800.0,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 280800000.0,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6000000000000.0,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6000000000000.0,
        g3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6000000000000.0,
        psi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.418e-13,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.2e-10,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta_v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        d_u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 24.0,
        d_u2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 400.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.G = G
        self.N = N
        self.AU = AU
        self.AV = AV
        self.V = V
        self.U = U
        self.r1 = r1
        self.r2 = r2
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.psi = psi
        self.alpha = alpha
        self.a4 = a4
        self.beta_v = beta_v
        self.d_u1 = d_u1
        self.d_u2 = d_u2
        self.u1 = u1


class Her2TumourElement(Element):
    """HER2+ breast cancer cells under trastuzumab and paclitaxel.

    The model includes antibody binding and drug uptake.

    The tumour grows logistically, `dA/dt = r (1 - DIm) (1 - A / (kc - DCh)) A`. The bound antibody `Ab` lowers the
    proliferation by `DIm = a1 Ab` from the time `t_a` on, and the internalized paclitaxel `Pin` lowers the carrying
    capacity by `DCh = sigma a2 exp(-gamma_p (t - t_p)) Pin` from the dose time `t_p` on. The free antibody and the
    free paclitaxel are drivers, held at the levels `antibody` and `paclitaxel` from their dose times on. The antibody
    binds the receptors, `dAb/dt = theta_ab (Af - Ab) (theta_r A - Ab)`, and the paclitaxel enters the cells,
    `dPin/dt = alpha_p (Pf - Pin) A`, while the free drugs are present. Cells are counted in millions, so the carrying
    capacity is 0.67, the antibody is in millions of molecules, and time is in days.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial number of cells.

        Ab:
            The initial bound antibody.

        Pin:
            The initial internalized paclitaxel.

        r:
            The growth rate.

        kc:
            The carrying capacity.

        a1:
            The inhibition of proliferation per unit of bound antibody.

        a2:
            The reduction of the carrying capacity per unit of internalized paclitaxel.

        gamma_p:
            The decay rate of the cytotoxicity of paclitaxel.

        sigma:
            The synergy of the two drugs, 1 for none.

        theta_ab:
            The binding rate of the antibody.

        theta_r:
            The number of receptors per cell.

        alpha_p:
            The uptake rate of paclitaxel.

        antibody:
            The level of free antibody from its dose on.

        paclitaxel:
            The level of free paclitaxel from its dose on, in pmol.

        t_antibody:
            The time of the antibody dose.

        t_a:
            The time from which the bound antibody inhibits proliferation.

        t_p:
            The time of the paclitaxel dose.

    Examples:
        >>> from skcomplex.systems.biology.oncology import Her2TumourElement
        >>> Her2TumourElement('tumour').kc
        0.67
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.035,
        Ab: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Pin: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.83,
        kc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.67,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.51,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 23.0,
        gamma_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6.35,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        theta_ab: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.78,
        theta_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1940000.0,
        alpha_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.68,
        antibody: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        paclitaxel: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        t_antibody: float = 1.0,
        t_a: float = 2.0,
        t_p: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.Ab = Ab
        self.Pin = Pin
        self.r = r
        self.kc = kc
        self.a1 = a1
        self.a2 = a2
        self.gamma_p = gamma_p
        self.sigma = sigma
        self.theta_ab = theta_ab
        self.theta_r = theta_r
        self.alpha_p = alpha_p
        self.antibody = antibody
        self.paclitaxel = paclitaxel
        self.t_antibody = t_antibody
        self.t_a = t_a
        self.t_p = t_p

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the tumour growth, the antibody binding, and the paclitaxel uptake.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, Ab, Pin, t = s.A, s.Ab, s.Pin, s.t
        free_antibody = where(t >= self.t_antibody, s.antibody, 0)
        free_paclitaxel = where(t >= self.t_p, s.paclitaxel, 0)
        inhibition = where(t > self.t_a, s.a1 * Ab, 0)
        cytotoxicity = where(t >= self.t_p, s.sigma * s.a2 * exp(-s.gamma_p * (t - self.t_p)) * Pin, 0)
        binding = s.theta_ab * (free_antibody - Ab) * (s.theta_r * A - Ab)
        return [
            RateLaw(A, s.r * (1 - inhibition) * (1 - A / (s.kc - cytotoxicity)) * A),
            RateLaw(Ab, where(free_antibody > 0, binding, 0)),
            RateLaw(Pin, where(free_paclitaxel > 0, s.alpha_p * (free_paclitaxel - Pin) * A, 0)),
        ]


def _toxicity(concentration: Expression, scale: Expression) -> Expression:
    """Return a toxicity linear in a low concentration and exponential in a high one.

    The toxicity is `exp(c / scale) - 1`.
    """
    return exp(concentration / scale) - 1


def _total_toxicity(s: SimpleNamespace) -> Expression:
    """Return the mean toxicity of the antiangiogenic and the immunotherapeutic drug."""
    tumour = s.tumour
    return (_toxicity(tumour.c_an, tumour.tox_an) + _toxicity(tumour.c_im, tumour.tox_im)) / 2


class ChemoimmunotherapySystem(BaseSystem):
    """A tumour with NK cells, CD8 T cells, and lymphocytes under therapy.

    The therapy is chemotherapy, IL-2, and TIL injection.

    The tumour, implanted in a mouse, grows toward a high equilibrium that the immune cells cannot hold back. The doses
    of chemotherapy shrink it, and the injected tumour-infiltrating lymphocytes clear what the drug leaves. The
    parameters are the table of the book, except the decay rates of the drugs, which the book's table swaps. The
    chemotherapeutic drug decays at 0.9 per day and IL-2 at 10 per day, as in de Pillis et al. 2006. The schedule of
    three weekly one-day doses from day 7 is a choice, since the book gives none. Cells are counted, and time is in
    days. The system has the states `tumour.A`, `tumour.N`, `tumour.E`, `tumour.C`, `tumour.I`, and `tumour.M`.

    Args:
        A:
            The initial number of tumour cells.

        N:
            The initial number of NK cells.

        E:
            The initial number of CD8 T cells.

        C:
            The initial number of circulating lymphocytes.

        u_til:
            The rate of TIL injection during a dose, in cells per day.

        u_il2:
            The rate of IL-2 injection during a dose.

        u_chemo:
            The rate of chemotherapy infusion during a dose.

        start:
            The time of the first dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ChemoimmunotherapySystem
        >>> ChemoimmunotherapySystem(u_til=0.0).prepare().system_.elements[0].u_til
        0.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'immunotherapy', 'combination therapy', 'ode', 'dosing')
    references = (BOOK, 'de Pillis, Gu, and Radunskaya 2006')
    until = 60.0
    dt = 0.2
    solver = ScipyODESolver(method='LSODA', rtol=1e-6, atol=1e-6)

    def __init__(
        self,
        A: float = 1e6,
        N: float = 5e4,
        E: float = 100.0,
        C: float = 1.1e7,
        *,
        u_til: float = 1e7,
        u_il2: float = 1e7,
        u_chemo: float = 2.0,
        start: float = 7.0,
        period: float = 7.0,
        width: float = 1.0,
        doses: int = 3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A = A
        self.N = N
        self.E = E
        self.C = C
        self.u_til = u_til
        self.u_il2 = u_il2
        self.u_chemo = u_chemo
        self.start = start
        self.period = period
        self.width = width
        self.doses = doses

    @staticmethod
    def _effector_kill(target: SimpleNamespace) -> Expression:
        """Return the ratio-dependent kill rate of tumour cells by effector cells.

        The rate is `D = a1 (E^l) / (g1 A^l + E^l)`.
        """
        ratio = abs(target.E) ** target.l
        kill: Expression = target.a1 * ratio / (target.g1 * abs(target.A) ** target.l + ratio)
        return kill

    def _build_system(self) -> System:
        """Return the parts of a tumour with NK cells."""
        tumour = ChemoimmunotherapyTumourElement(
            'tumour',
            A=self.A,
            N=self.N,
            E=self.E,
            C=self.C,
            u_til=self.u_til,
            u_il2=self.u_il2,
            u_chemo=self.u_chemo,
            start=self.start,
            period=self.period,
            width=self.width,
            doses=self.doses,
        )
        interactions = [
            RateInteraction(
                'tumour_growth',
                target='tumour',
                state='A',
                rate=lambda s: s.target.r1 * s.target.A * (1 - s.target.b1 * s.target.A),
            ),
            ReactionInteraction(
                'nk_kill',
                target='tumour',
                reactants={'A': 1},
                modifiers={'N': 1},
                propensity=lambda s: s.target.c1 * s.target.N * s.target.A,
            ),
            ReactionInteraction(
                'effector_kill',
                target='tumour',
                reactants={'A': 1},
                propensity=lambda s: self._effector_kill(s.target) * s.target.A,
            ),
            ReactionInteraction(
                'chemo_tumour_kill',
                target='tumour',
                reactants={'A': 1},
                propensity=lambda s: s.target.a2 * (1 - exp(-s.target.M)) * s.target.A,
            ),
            RateInteraction('nk_production', target='tumour', state='N', rate=lambda s: s.target.m1 * s.target.C),
            ReactionInteraction(
                'nk_death',
                target='tumour',
                reactants={'N': 1},
                propensity=lambda s: s.target.d1 * s.target.N,
            ),
            RateInteraction(
                'nk_recruitment',
                target='tumour',
                state='N',
                rate=lambda s: s.target.r2 * s.target.A**2 / (s.target.g2 + s.target.A**2) * s.target.N,
            ),
            ReactionInteraction(
                'nk_inactivation',
                target='tumour',
                reactants={'N': 1},
                modifiers={'A': 1},
                propensity=lambda s: s.target.psi1 * s.target.N * s.target.A,
            ),
            ReactionInteraction(
                'chemo_nk_kill',
                target='tumour',
                reactants={'N': 1},
                propensity=lambda s: s.target.a3 * (1 - exp(-s.target.M)) * s.target.N,
            ),
            RateInteraction(
                'effector_recruitment',
                target='tumour',
                state='E',
                rate=lambda s: s.target.r3
                * (self._effector_kill(s.target) * s.target.A) ** 2
                / (s.target.g3 + (self._effector_kill(s.target) * s.target.A) ** 2)
                * s.target.E,
            ),
            ReactionInteraction(
                'effector_death',
                target='tumour',
                reactants={'E': 1},
                propensity=lambda s: s.target.d2 * s.target.E,
            ),
            ReactionInteraction(
                'effector_inactivation',
                target='tumour',
                reactants={'E': 1},
                modifiers={'A': 1},
                propensity=lambda s: s.target.psi2 * s.target.E * s.target.A,
            ),
            ReactionInteraction(
                'nk_regulation',
                target='tumour',
                reactants={'E': 1},
                propensity=lambda s: s.target.psi3 * s.target.N * s.target.E**2,
            ),
            ReactionInteraction(
                'chemo_effector_kill',
                target='tumour',
                reactants={'E': 1},
                propensity=lambda s: s.target.a4 * (1 - exp(-s.target.M)) * s.target.E,
            ),
            RateInteraction(
                'effector_activation',
                target='tumour',
                state='E',
                rate=lambda s: (s.target.r4 * s.target.N + s.target.r5 * s.target.C) * s.target.A,
            ),
            MichaelisMentenInteraction(
                'il2_proliferation',
                target='tumour',
                products={'E': 1},
                substrate='I',
                vmax=lambda s: s.target.r6,
                km=lambda s: s.target.g4,
                modifiers={'E': 1},
            ),
            DosingInteraction(
                'til_infusion',
                target='tumour',
                state='E',
                dose=lambda s: s.target.u_til,
                period=tumour.period,
                width=tumour.width,
                start=tumour.start,
                count=tumour.doses,
            ),
            RateInteraction('lymphocyte_production', target='tumour', state='C', rate=lambda s: s.target.s1),
            ReactionInteraction(
                'lymphocyte_death',
                target='tumour',
                reactants={'C': 1},
                propensity=lambda s: s.target.d3 * s.target.C,
            ),
            ReactionInteraction(
                'chemo_lymphocyte_kill',
                target='tumour',
                reactants={'C': 1},
                propensity=lambda s: s.target.a5 * (1 - exp(-s.target.M)) * s.target.C,
            ),
            ReactionInteraction(
                'il2_clearance',
                target='tumour',
                reactants={'I': 1},
                propensity=lambda s: s.target.d_u1 * s.target.I,
            ),
            DosingInteraction(
                'il2_infusion',
                target='tumour',
                state='I',
                dose=lambda s: s.target.u_il2,
                period=tumour.period,
                width=tumour.width,
                start=tumour.start,
                count=tumour.doses,
            ),
            ReactionInteraction(
                'chemo_clearance',
                target='tumour',
                reactants={'M': 1},
                propensity=lambda s: s.target.d_u2 * s.target.M,
            ),
            DosingInteraction(
                'chemo_infusion',
                target='tumour',
                state='M',
                dose=lambda s: s.target.u_chemo,
                period=tumour.period,
                width=tumour.width,
                start=tumour.start,
                count=tumour.doses,
            ),
        ]
        return System(self.label, elements=[tumour], interactions=interactions)


class ClonalChemoimmunotherapySystem(BaseSystem):
    """Two competing tumour clones under a drug and a vaccine.

    One clone is sensitive and the other resistant to chemotherapy.

    The chemotherapy kills the sensitive first clone ten times faster than the resistant second clone. Alone, it removes
    the sensitive clone and frees the resistant clone from competition, so the tumour relapses resistant. With the
    dendritic-cell vaccine, the recruited immune cells clear both clones. The book does not system the drugs, so their
    effects are constant rates, and their values are a choice. The competition rates take a value in the book's range of
    0.01 to 1. Cells are counted, and time is in days. The system has the clones `tumour.A1` and `tumour.A2` and the
    immune cells `tumour.E`.

    Args:
        A1:
            The initial number of cells of the sensitive clone.

        A2:
            The initial number of cells of the resistant clone.

        E:
            The initial number of immune cells.

        c1:
            The competition of the second clone on the first.

        c3:
            The competition of the first clone on the second.

        u_chemo1:
            The chemotherapy kill rate of the first clone.

        u_chemo2:
            The chemotherapy kill rate of the second clone.

        u_vaccine:
            The level of the immunotherapeutic agent.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ClonalChemoimmunotherapySystem
        >>> ClonalChemoimmunotherapySystem(u_vaccine=0.0).prepare().system_.elements[0].u_vaccine
        0.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'immunotherapy', 'combination therapy', 'heterogeneity', 'resistance', 'ode')
    references = (BOOK, 'Piretto, Delitala, and Ferraro 2018')
    until = 300.0
    dt = 1.0

    def __init__(
        self,
        A1: float = 9e4,
        A2: float = 4e4,
        E: float = 500.0,
        *,
        c1: float = 0.1,
        c3: float = 0.1,
        u_chemo1: float = 0.2,
        u_chemo2: float = 0.02,
        u_vaccine: float = 20.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A1 = A1
        self.A2 = A2
        self.E = E
        self.c1 = c1
        self.c3 = c3
        self.u_chemo1 = u_chemo1
        self.u_chemo2 = u_chemo2
        self.u_vaccine = u_vaccine

    def _build_system(self) -> System:
        """Return the parts of two competing tumour clones."""
        tumour = ClonalTumourElement(
            'tumour',
            A1=self.A1,
            A2=self.A2,
            E=self.E,
            c1=self.c1,
            c3=self.c3,
            u_chemo1=self.u_chemo1,
            u_chemo2=self.u_chemo2,
            u_vaccine=self.u_vaccine,
        )
        interactions = [
            RateInteraction(
                'first_growth',
                target='tumour',
                state='A1',
                rate=lambda s: s.target.r1 * s.target.A1 * (1 - s.target.b1 * s.target.A1),
            ),
            ReactionInteraction(
                'first_competition',
                target='tumour',
                reactants={'A1': 1},
                modifiers={'A2': 1},
                propensity=lambda s: s.target.c1 * s.target.A2 * s.target.b1 * s.target.A1,
            ),
            ReactionInteraction(
                'first_immune_kill',
                target='tumour',
                reactants={'A1': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.c2 * s.target.E * s.target.b1 * s.target.A1,
            ),
            ReactionInteraction(
                'first_chemotherapy',
                target='tumour',
                reactants={'A1': 1},
                propensity=lambda s: s.target.u_chemo1 * s.target.A1,
            ),
            ReactionInteraction(
                'first_vaccine_kill',
                target='tumour',
                reactants={'A1': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.b1 * s.target.u_vaccine * s.target.A1 * s.target.E,
            ),
            RateInteraction(
                'second_growth',
                target='tumour',
                state='A2',
                rate=lambda s: s.target.r2 * s.target.A2 * (1 - s.target.b2 * s.target.A2),
            ),
            ReactionInteraction(
                'second_competition',
                target='tumour',
                reactants={'A2': 1},
                modifiers={'A1': 1},
                propensity=lambda s: s.target.c3 * s.target.A1 * s.target.b2 * s.target.A2,
            ),
            ReactionInteraction(
                'second_immune_kill',
                target='tumour',
                reactants={'A2': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.c4 * s.target.E * s.target.b2 * s.target.A2,
            ),
            ReactionInteraction(
                'second_chemotherapy',
                target='tumour',
                reactants={'A2': 1},
                propensity=lambda s: s.target.u_chemo2 * s.target.A2,
            ),
            ReactionInteraction(
                'second_vaccine_kill',
                target='tumour',
                reactants={'A2': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.b2 * s.target.u_vaccine * s.target.A2 * s.target.E,
            ),
            RateInteraction(
                'immune_growth',
                target='tumour',
                state='E',
                rate=lambda s: s.target.r3 * s.target.E * (1 - s.target.E / s.target.theta),
            ),
            RateInteraction(
                'immune_stimulation',
                target='tumour',
                state='E',
                rate=lambda s: (s.target.s1 * s.target.A1 + s.target.s2 * s.target.A2) * s.target.E / s.target.theta,
            ),
        ]
        return System(self.label, elements=[tumour], interactions=interactions)


class ChemovirotherapySystem(BaseSystem):
    """Uninfected and infected tumour cells, an oncolytic virus, and a drug.

    The drug is chemotherapeutic.

    The virus infects the growing tumour, and the drug kills both uninfected and infected cells. Over the default run of
    64 days, before the virus bursts through the tumour, the combination leaves the smallest tumour. Later the virus
    alone clears the tumour slightly faster, because the drug also kills the infected cells that make the virus. The
    infusion rate is a choice, since the book gives none. One unit of cells is a million cells, and one unit of time is
    two days. The system has the cells `tumour.AU` and `tumour.AV`, the virus `tumour.V`, and the drug `tumour.U`.

    Args:
        AU:
            The initial number of uninfected cells.

        V:
            The initial number of virions, 0 for chemotherapy alone.

        U:
            The initial amount of drug.

        u1:
            The infusion rate of the drug, 0 for virotherapy alone.

        infusion:
            The form of the infusion, `'constant'`, `'exponential'`, or `'periodic'`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ChemovirotherapySystem
        >>> ChemovirotherapySystem(infusion='exponential').prepare().system_.elements[0].infusion
        'exponential'
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'virotherapy', 'combination therapy', 'oncolytic virus', 'ode')
    references = (BOOK, 'Malinzi 2019')
    until = 32.0
    dt = 0.1

    def __init__(
        self,
        AU: float = 1.0,
        V: float = 0.1,
        U: float = 0.1,
        *,
        u1: float = 10.0,
        infusion: str = 'constant',
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.AU = AU
        self.V = V
        self.U = U
        self.u1 = u1
        self.infusion = infusion

    @staticmethod
    def _infusion_shape(s: SimpleNamespace, infusion: str) -> Expression:
        """Return the shape in time of an infusion.

        The infusion is constant, exponentially decaying, or periodic.
        """
        shapes = {'constant': 1, 'exponential': exp(-s.target.d_u * s.t), 'periodic': sin(s.target.d_u * s.t) ** 2}
        shape: Expression = shapes[infusion]
        return shape

    def _build_system(self) -> System:
        """Return the parts of tumour cells, an oncolytic virus, and a drug.

        The tumour cells are uninfected and infected, and the drug is chemotherapeutic.
        """
        tumour = ChemovirotherapyTumourElement(
            'tumour',
            AU=self.AU,
            V=self.V,
            U=self.U,
            u1=self.u1,
            infusion=self.infusion,
        )
        interactions = [
            RateInteraction(
                'tumour_growth',
                target='tumour',
                state='AU',
                rate=lambda s: s.target.r1 * s.target.AU * (1 - s.target.b * (s.target.AU + s.target.AV)),
            ),
            ReactionInteraction(
                'infection',
                target='tumour',
                reactants={'AU': 1, 'V': 1},
                products={'AV': 1},
                propensity=lambda s: s.target.alpha * s.target.AU * s.target.V,
            ),
            ReactionInteraction(
                'drug_kill',
                target='tumour',
                reactants={'AU': 1},
                modifiers={'U': 1},
                propensity=lambda s: s.target.a1 * s.target.AU * s.target.U,
            ),
            ReactionInteraction(
                'lysis',
                target='tumour',
                reactants={'AV': 1},
                propensity=lambda s: s.target.a2 * s.target.AV,
            ),
            ReactionInteraction(
                'infected_drug_kill',
                target='tumour',
                reactants={'AV': 1},
                modifiers={'U': 1},
                propensity=lambda s: s.target.a3 * s.target.AV * s.target.U,
            ),
            RateInteraction(
                'burst',
                target='tumour',
                state='V',
                rate=lambda s: s.target.beta_v * s.target.a2 * s.target.AV,
            ),
            ReactionInteraction(
                'virus_clearance',
                target='tumour',
                reactants={'V': 1},
                propensity=lambda s: s.target.d1 * s.target.V,
            ),
            ReactionInteraction(
                'drug_clearance',
                target='tumour',
                reactants={'U': 1},
                propensity=lambda s: s.target.d_u * s.target.U,
            ),
            RateInteraction(
                'infusion',
                target='tumour',
                state='U',
                rate=lambda s: s.target.u1 * self._infusion_shape(s, self.infusion),
            ),
        ]
        return System(self.label, elements=[tumour], interactions=interactions)


class GliomaChemovirotherapySystem(BaseSystem):
    """Glia, neurons, and a glioma under temozolomide and an oncolytic virus.

    The temozolomide infusion is constant.

    Untreated, the glioma grows and crowds out the glia, and the neurons die with the glia. The virus or the drug clears
    the glioma, and the neurons survive. The drug also kills some neurons. The book gives no initial virus or drug, so
    the run starts with no drug and with a billion virions. Cells and virions are counted, the drug is in mg per m², and
    time is in days. The system has the glia `brain.G`, the neurons `brain.N`, the glioma `brain.AU` and `brain.AV`, the
    virus `brain.V`, and the drug `brain.U`.

    Args:
        AU:
            The initial number of glioma cells.

        V:
            The initial number of virions, 0 for chemotherapy alone.

        u1:
            The infusion rate of temozolomide, in mg per m² per day, 0 for virotherapy alone.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import GliomaChemovirotherapySystem
        >>> GliomaChemovirotherapySystem(u1=0.0).prepare().system_.elements[0].u1
        0.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'virotherapy', 'combination therapy', 'glioma', 'neurons', 'ode')
    references = (BOOK, 'Urenda-Cázares, Gallegos, and Macías-Díaz 2020', 'Iarosz et al. 2015')
    until = 300.0
    dt = 1.0

    def __init__(
        self,
        AU: float = 6e10,
        V: float = 1e9,
        *,
        u1: float = 400.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.AU = AU
        self.V = V
        self.u1 = u1

    @staticmethod
    def _glia_rate(target: SimpleNamespace) -> Expression:
        """Return the net rate of the glia, whose decline kills neurons."""
        G = target.G
        rate: Expression = (
            target.r1 * G * (1 - target.b * G) - target.c1 * G * target.AU - target.a1 * G * target.U / (target.g1 + G)
        )
        return rate

    def _build_system(self) -> System:
        """Return the parts of the system."""
        brain = GliomaBrainElement('brain', AU=self.AU, V=self.V, u1=self.u1)
        interactions = [
            RateInteraction(
                'glia_growth',
                target='brain',
                state='G',
                rate=lambda s: s.target.r1 * s.target.G * (1 - s.target.b * s.target.G),
            ),
            ReactionInteraction(
                'glia_competition',
                target='brain',
                reactants={'G': 1},
                modifiers={'AU': 1},
                propensity=lambda s: s.target.c1 * s.target.G * s.target.AU,
            ),
            MichaelisMentenInteraction(
                'glia_drug_kill',
                target='brain',
                reactants={'G': 1},
                substrate='G',
                vmax=lambda s: s.target.a1,
                km=lambda s: s.target.g1,
                modifiers={'U': 1},
            ),
            ReactionInteraction(
                'neuron_loss',
                target='brain',
                reactants={'N': 1},
                propensity=lambda s: -s.target.psi
                * where(self._glia_rate(s.target) < 0, self._glia_rate(s.target), 0)
                * s.target.N,
            ),
            MichaelisMentenInteraction(
                'neuron_drug_kill',
                target='brain',
                reactants={'N': 1},
                substrate='N',
                vmax=lambda s: s.target.a2,
                km=lambda s: s.target.g2,
                modifiers={'U': 1},
            ),
            RateInteraction(
                'glioma_growth',
                target='brain',
                state='AU',
                rate=lambda s: s.target.r2 * s.target.AU * (1 - s.target.b * (s.target.AU + s.target.AV)),
            ),
            ReactionInteraction(
                'glioma_competition',
                target='brain',
                reactants={'AU': 1},
                modifiers={'G': 1},
                propensity=lambda s: s.target.c2 * s.target.G * s.target.AU,
            ),
            MichaelisMentenInteraction(
                'glioma_drug_kill',
                target='brain',
                reactants={'AU': 1},
                substrate='AU',
                vmax=lambda s: s.target.a3,
                km=lambda s: s.target.g3,
                modifiers={'U': 1},
            ),
            ReactionInteraction(
                'infection',
                target='brain',
                reactants={'AU': 1, 'V': 1},
                products={'AV': 1},
                propensity=lambda s: s.target.alpha * s.target.AU * s.target.V,
            ),
            ReactionInteraction(
                'lysis',
                target='brain',
                reactants={'AV': 1},
                propensity=lambda s: s.target.a4 * s.target.AV,
            ),
            RateInteraction(
                'burst',
                target='brain',
                state='V',
                rate=lambda s: s.target.beta_v * s.target.a4 * s.target.AV,
            ),
            ReactionInteraction(
                'virus_clearance',
                target='brain',
                reactants={'V': 1},
                propensity=lambda s: s.target.d_u1 * s.target.V,
            ),
            ReactionInteraction(
                'drug_clearance',
                target='brain',
                reactants={'U': 1},
                propensity=lambda s: s.target.d_u2 * s.target.U,
            ),
            DosingInteraction('infusion', target='brain', state='U', dose=lambda s: s.target.u1),
        ]
        return System(self.label, elements=[brain], interactions=interactions)


class HER2TargetedTherapySystem(BaseSystem):
    """HER2+ breast cancer cells under trastuzumab and paclitaxel.

    The model includes antibody binding and drug uptake.

    The free antibody and the free paclitaxel are drivers held at their levels from their doses on. The antibody binds
    the receptors, which are in excess, so the bound antibody rises to the free antibody, and from the time `t_a` on it
    lowers the growth rate to `r (1 - a1 Ab)`. The paclitaxel enters the cells and briefly lowers the carrying capacity.
    The cytotoxicity falls from one at the dose, so `t_p` is the dose time. The book's units cannot hold 35,000 cells
    with a capacity of 0.67, so cells are counted in millions and the antibody in millions of molecules. The dose levels
    and times are a choice. Time is in days. The system has the cells `tumour.A`, the bound antibody `tumour.Ab`, and
    the internalized paclitaxel `tumour.Pin`.

    Args:
        A:
            The initial number of cells, in millions.

        antibody:
            The level of free antibody from its dose on.

        paclitaxel:
            The level of free paclitaxel from its dose on, in pmol.

        t_antibody:
            The time of the antibody dose.

        t_a:
            The time from which the bound antibody inhibits proliferation.

        t_p:
            The time of the paclitaxel dose.

        sigma:
            The synergy of the two drugs, 1 for none.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import HER2TargetedTherapySystem
        >>> HER2TargetedTherapySystem(sigma=1.5).prepare().system_.elements[0].sigma
        1.5
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'targeted therapy', 'combination therapy', 'HER2', 'antibody', 'ode')
    references = (BOOK, 'Jarrett et al. 2019')
    until = 10.0
    dt = 0.02
    solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)

    def __init__(
        self,
        A: float = 0.035,
        *,
        antibody: float = 0.2,
        paclitaxel: float = 2.0,
        t_antibody: float = 1.0,
        t_a: float = 2.0,
        t_p: float = 1.0,
        sigma: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A = A
        self.antibody = antibody
        self.paclitaxel = paclitaxel
        self.t_antibody = t_antibody
        self.t_a = t_a
        self.t_p = t_p
        self.sigma = sigma

    def _build_system(self) -> System:
        """Return the parts of HER2+ breast cancer cells under trastuzumab and paclitaxel."""
        tumour = Her2TumourElement(
            'tumour',
            A=self.A,
            antibody=self.antibody,
            paclitaxel=self.paclitaxel,
            t_antibody=self.t_antibody,
            t_a=self.t_a,
            t_p=self.t_p,
            sigma=self.sigma,
        )
        return System(self.label, elements=[tumour])


class AngiogenesisImmunotherapySystem(BaseSystem):
    """The Hahnfeldt tumour with effector cells under bevacizumab and atezolizumab.

    The drugs are given every three weeks.

    Each drug alone holds the tumour or slows it, and together they shrink it. The drug amounts follow the
    two-compartment system whose bolus solution is the book's closed form, with the elimination rate `k10 = alpha beta /
    k21`. The effector cells grow as `r1 (A - c1 A^2) E`, so `1 / c1` is the volume above which the tumour suppresses
    them. The drug effects in the vasculature and effector balances are the saturating effects of the book's
    pharmacodynamics. The resistance capacity `delta_r = 0.01` and its threshold `u_theta = 5` mg per kg are a choice,
    since the book gives none. The toxicity of each drug is `exp(c / tox) - 1`, linear for a low amount and exponential
    for a high one, and the observable `toxicity` is the mean of both. Volumes are in mm³, drug amounts in mg per kg,
    and time is in days. The system has the tumour `tumour.A`, the capacity `tumour.Vb`, the effector cells `tumour.E`,
    the drug amounts `tumour.c_an` and `tumour.c_im`, and the observable `toxicity`.

    Args:
        A:
            The initial tumour volume.

        Vb:
            The initial vascular capacity.

        E:
            The initial effector cells, dimensionless.

        dose_an:
            The dose of bevacizumab, in mg per kg, 0 for immunotherapy alone.

        dose_im:
            The dose of atezolizumab, in mg per kg, 0 for antiangiogenic therapy alone.

        a2:
            The highest destruction rate of the vasculature by bevacizumab, in the book's range of 1.35 to 11.25.

        a3:
            The highest boost of effector cells by atezolizumab, in the book's range of 0.1 to 0.5.

        period:
            The time between doses.

        doses:
            The number of doses.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import AngiogenesisImmunotherapySystem
        >>> AngiogenesisImmunotherapySystem(dose_an=0.0).prepare().system_.elements[0].dose_an
        0.0
    """

    field = 'biology/oncology'
    topics = ('antiangiogenic therapy', 'immunotherapy', 'combination therapy', 'pharmacokinetics', 'resistance', 'ode')
    references = (BOOK, 'Teles and Lemos 2019', 'Hahnfeldt et al. 1999')
    until = 150.0
    dt = 0.25
    observables = (ObservableSpec('toxicity', _total_toxicity),)

    def __init__(
        self,
        A: float = 5000.0,
        Vb: float = 625.0,
        E: float = 0.1,
        *,
        dose_an: float = 15.0,
        dose_im: float = 20.0,
        a2: float = 1.35,
        a3: float = 0.1,
        period: float = 21.0,
        doses: int = 6,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A = A
        self.Vb = Vb
        self.E = E
        self.dose_an = dose_an
        self.dose_im = dose_im
        self.a2 = a2
        self.a3 = a3
        self.period = period
        self.doses = doses

    @staticmethod
    def _elimination(target: SimpleNamespace, drug: str) -> Expression:
        """Return the elimination rate `k10 = alpha beta / k21` of a drug.

        The drug is eliminated from its central compartment.
        """
        alpha, beta, k21 = (getattr(target, f'{name}_{drug}') for name in ('alpha', 'beta', 'k21'))
        elimination: Expression = alpha * beta / k21
        return elimination

    @staticmethod
    def _drug_effect(target: SimpleNamespace, drug: str) -> Expression:
        """Return the Emax effect of a drug.

        The resistance shifts its half-effect concentration.
        """
        c, R, u50 = (getattr(target, f'{name}_{drug}') for name in ('c', 'R', 'u50'))
        effect: Expression = c**target.g / (((1 + target.delta_r * R) * u50) ** target.g + c**target.g)
        return effect

    def _build_system(self) -> System:
        """Return the parts of the Hahnfeldt tumour."""
        tumour = AngiogenesisImmunotherapyTumourElement(
            'tumour',
            A=self.A,
            Vb=self.Vb,
            E=self.E,
            dose_an=self.dose_an,
            dose_im=self.dose_im,
            a2=self.a2,
            a3=self.a3,
            period=self.period,
            doses=self.doses,
        )
        interactions = [
            RateInteraction(
                'tumour_growth',
                target='tumour',
                state='A',
                rate=lambda s: -s.target.v1 * s.target.A * log(s.target.A / s.target.Vb),
            ),
            ReactionInteraction(
                'immune_kill',
                target='tumour',
                reactants={'A': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.a1 * s.target.A * s.target.E,
            ),
            ReactionInteraction(
                'vessel_loss',
                target='tumour',
                reactants={'Vb': 1},
                propensity=lambda s: s.target.v2 * s.target.Vb,
            ),
            RateInteraction('stimulation', target='tumour', state='Vb', rate=lambda s: s.target.xi_s * s.target.A),
            ReactionInteraction(
                'inhibition',
                target='tumour',
                reactants={'Vb': 1},
                propensity=lambda s: s.target.xi_i * s.target.Vb * s.target.A ** (2 / 3),
            ),
            ReactionInteraction(
                'pruning',
                target='tumour',
                reactants={'Vb': 1},
                propensity=lambda s: s.target.a2 * s.target.Vb * self._drug_effect(s.target, 'an'),
            ),
            RateInteraction(
                'immune_growth',
                target='tumour',
                state='E',
                rate=lambda s: s.target.r1 * (s.target.A - s.target.c1 * s.target.A**2) * s.target.E,
            ),
            ReactionInteraction(
                'immune_death',
                target='tumour',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            RateInteraction('immune_source', target='tumour', state='E', rate=lambda s: s.target.s_e),
            RateInteraction(
                'immune_boost',
                target='tumour',
                state='E',
                rate=lambda s: s.target.a3 * s.target.E * self._drug_effect(s.target, 'im'),
            ),
            DosingInteraction(
                'dosing_an',
                target='tumour',
                state='c_an',
                dose=lambda s: s.target.dose_an / tumour.width,
                period=tumour.period,
                width=tumour.width,
                count=tumour.doses,
            ),
            ReactionInteraction(
                'elimination_an',
                target='tumour',
                reactants={'c_an': 1},
                propensity=lambda s: self._elimination(s.target, 'an') * s.target.c_an,
            ),
            ReactionInteraction(
                'distribution_an',
                target='tumour',
                reactants={'c_an': 1},
                products={'q_an': 1},
                propensity=lambda s: s.target.k12_an * s.target.c_an,
            ),
            ReactionInteraction(
                'redistribution_an',
                target='tumour',
                reactants={'q_an': 1},
                products={'c_an': 1},
                propensity=lambda s: s.target.k21_an * s.target.q_an,
            ),
            RateInteraction(
                'resistance_an',
                target='tumour',
                state='R_an',
                rate=lambda s: where(s.target.c_an < s.target.u_theta, s.target.u_theta - s.target.c_an, 0),
            ),
            DosingInteraction(
                'dosing_im',
                target='tumour',
                state='c_im',
                dose=lambda s: s.target.dose_im / tumour.width,
                period=tumour.period,
                width=tumour.width,
                count=tumour.doses,
            ),
            ReactionInteraction(
                'elimination_im',
                target='tumour',
                reactants={'c_im': 1},
                propensity=lambda s: self._elimination(s.target, 'im') * s.target.c_im,
            ),
            ReactionInteraction(
                'distribution_im',
                target='tumour',
                reactants={'c_im': 1},
                products={'q_im': 1},
                propensity=lambda s: s.target.k12_im * s.target.c_im,
            ),
            ReactionInteraction(
                'redistribution_im',
                target='tumour',
                reactants={'q_im': 1},
                products={'c_im': 1},
                propensity=lambda s: s.target.k21_im * s.target.q_im,
            ),
            RateInteraction(
                'resistance_im',
                target='tumour',
                state='R_im',
                rate=lambda s: where(s.target.c_im < s.target.u_theta, s.target.u_theta - s.target.c_im, 0),
            ),
        ]
        return System(self.label, elements=[tumour], interactions=interactions)
