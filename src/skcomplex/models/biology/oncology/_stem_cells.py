"""Oncology models: normal, malignant, and leukaemic stem cells and the lineages they feed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, Literal

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp, log, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import delayed, pulse
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class StemCellDifferentiationElement(Element):
    """Normal and malignant lineages of quiescent stem cells, progenitors, basal cells, and differentiated cells.

    In each lineage the quiescent stem cells become progenitors at the rate `k12` and progenitors return at the rate
    `k21 (1 - NPr / (NPr + NPinf))`. Progenitors renew logistically, become basal cells, and basal cells
    differentiate. The renewal of normal progenitors gets the feedback `beta0 (1 - ND / (ND + kappa0))` of the
    differentiated cells. Normal progenitors mutate into malignant ones at the rate `m1`. A therapy multiplies the
    death rates of the progenitors and the basal cells by its dose, and that of the normal progenitors also by
    `1 + delta` times their renewal rate. A pulsed therapy multiplies them by 4 more from `start` for `duration`.

    Args:
        label:
            The label of the tissue.

        therapy:
            The mode of therapy, `'none'`, `'pulsed'`, or `'continuous'`.

        start:
            The time the pulse of a pulsed therapy starts.

        duration:
            The duration of the pulse of a pulsed therapy.

        **values:
            The values of the declared variables, by name:

            NSq:
                The initial normal quiescent stem cells.

            NPr:
                The initial normal progenitors.

            NBs:
                The initial normal basal cells.

            ND:
                The initial normal differentiated cells.

            ASq:
                The initial malignant quiescent stem cells.

            APr:
                The initial malignant progenitors.

            ABs:
                The initial malignant basal cells.

            AD:
                The initial malignant differentiated cells.

            d1:
                The death rate of normal quiescent stem cells.

            d2:
                The death rate of normal progenitors.

            d3:
                The death rate of normal basal cells.

            d4:
                The death rate of normal differentiated cells.

            d5:
                The death rate of malignant quiescent stem cells.

            d6:
                The death rate of malignant progenitors.

            d7:
                The death rate of malignant basal cells.

            d8:
                The death rate of malignant differentiated cells.

            k12:
                The rate at which normal quiescent stem cells become progenitors.

            k21:
                The largest rate at which normal progenitors return to quiescence.

            k23:
                The rate at which normal progenitors become basal cells.

            k34:
                The rate at which normal basal cells differentiate.

            k56:
                The rate at which malignant quiescent stem cells become progenitors.

            k65:
                The largest rate at which malignant progenitors return to quiescence.

            k67:
                The rate at which malignant progenitors become basal cells.

            k78:
                The rate at which malignant basal cells differentiate.

            r1:
                The renewal rate of normal progenitors.

            r2:
                The renewal rate of normal basal cells.

            r3:
                The renewal rate of malignant progenitors.

            r4:
                The renewal rate of malignant basal cells.

            m1:
                The mutation rate of normal progenitors.

            NPinf:
                The saturation constant of the return of normal progenitors.

            APinf:
                The saturation constant of the return of malignant progenitors.

            NPmax:
                The largest number of normal progenitors.

            APmax:
                The largest number of malignant progenitors.

            Nbmax:
                The largest number of normal basal cells.

            Abmax:
                The largest number of malignant basal cells.

            beta0:
                The feedback constant.

            kappa0:
                The number of differentiated cells that halves the feedback.

            delta:
                The weight of the renewal in the death the therapy causes to normal progenitors.

            up:
                The dose of the pulsed therapy.

            uc:
                The dose of the continuous therapy.

    Examples:
        >>> from skcomplex.models.biology.oncology import StemCellDifferentiationElement
        >>> StemCellDifferentiationElement('tissue', therapy='pulsed').therapy
        'pulsed'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        NSq: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        NPr: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        NBs: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        ND: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        ASq: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        APr: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ABs: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        d4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        d6: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        d7: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        d8: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        k12: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k21: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        k23: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k34: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
        k56: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k65: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        k67: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        k78: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        NPinf: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        APinf: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        NPmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        APmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 50000.0,
        Nbmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        Abmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5000.0,
        beta0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        kappa0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2000.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        up: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        uc: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        therapy: str = 'continuous',
        start: float = 100.0,
        duration: float = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.NSq = NSq
        self.NPr = NPr
        self.NBs = NBs
        self.ND = ND
        self.ASq = ASq
        self.APr = APr
        self.ABs = ABs
        self.AD = AD
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.d5 = d5
        self.d6 = d6
        self.d7 = d7
        self.d8 = d8
        self.k12 = k12
        self.k21 = k21
        self.k23 = k23
        self.k34 = k34
        self.k56 = k56
        self.k65 = k65
        self.k67 = k67
        self.k78 = k78
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.m1 = m1
        self.NPinf = NPinf
        self.APinf = APinf
        self.NPmax = NPmax
        self.APmax = APmax
        self.Nbmax = Nbmax
        self.Abmax = Abmax
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.delta = delta
        self.up = up
        self.uc = uc
        self.therapy = therapy
        self.start = start
        self.duration = duration

    def _therapy_deaths(self, s: SimpleNamespace, renewal: Expression) -> tuple[Expression, Expression]:
        """Return the death rate of normal progenitors and the factor of the other death rates under the therapy."""
        if self.therapy == 'continuous':
            return s.d2 * s.uc * (1 + s.delta * renewal), s.uc
        if self.therapy == 'pulsed':
            on = pulse(s.t, self.start, self.duration)
            return s.d2 * s.up * (1 + on * s.delta * renewal), s.up * (1 + 3 * on)
        return s.d2, 1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the transitions, renewals, deaths, and mutation of the eight compartments.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        NSq, NPr, NBs, ND, ASq, APr, ABs, AD = s.NSq, s.NPr, s.NBs, s.ND, s.ASq, s.APr, s.ABs, s.AD
        renewal = (s.r1 + s.beta0 * (1 - ND / (ND + s.kappa0))) * (1 - NPr / s.NPmax)
        d2, other_factor = self._therapy_deaths(s, renewal)
        normal_return = s.k21 * (1 - NPr / (NPr + s.NPinf)) * NPr
        malignant_return = s.k65 * (1 - APr / (APr + s.APinf)) * APr
        return [
            RateLaw(NSq, -s.d1 * NSq - s.k12 * NSq + normal_return),
            RateLaw(NPr, (renewal - d2 - s.m1 - s.k23) * NPr + s.k12 * NSq - normal_return),
            RateLaw(NBs, -s.d3 * other_factor * NBs + s.k23 * NPr - s.k34 * NBs + s.r2 * (1 - NBs / s.Nbmax) * NBs),
            RateLaw(ND, -s.d4 * ND + s.k34 * NBs),
            RateLaw(ASq, -s.d5 * ASq - s.k56 * ASq + malignant_return),
            RateLaw(
                APr,
                (s.r3 * (1 - APr / s.APmax) - s.d6 * other_factor - s.k67) * APr
                + s.k56 * ASq
                - malignant_return
                + s.m1 * NPr,
            ),
            RateLaw(ABs, -s.d7 * other_factor * ABs + s.k67 * APr - s.k78 * ABs + s.r4 * (1 - ABs / s.Abmax) * ABs),
            RateLaw(AD, -s.d8 * AD + s.k78 * ABs),
        ]


class BoneMarrowBloodElement(Element):
    """Normal and malignant cells in the bone marrow and the peripheral blood, and cancer stem cells.

    Normal marrow cells renew as `r1 NBM log((b1 + b2) / (theta1 + NBM))`, mutate into blasts at the rate `m1` and
    into cancer stem cells at the rate `m3`, and move to the blood, where they arrive after the maturation time
    `tau1` with the survival `exp(-alpha1 tau1 - r2 APB(t - tau1))`. Blasts grow as `r3 ABM log(b3 / (theta3 + ABM))`
    and move to the blood at the rate `m4`. Cancer stem cells renew logistically and convert into blasts after the
    time `tau2`. Malignant cells inhibit normal ones in both places.

    Args:
        label:
            The label of the patient.

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the renewal, mutation, migration, and delayed maturation and conversion of the five populations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        NBM, NPB, ABM, APB, AS = s.NBM, s.NPB, s.ABM, s.APB, s.AS
        release = s.m2 * NBM * exp(-s.r2 * APB) / (s.theta2 + NPB)
        survival = exp(-s.alpha1 * s.tau1 - s.r2 * delayed(APB, s.tau1))
        arrival = s.m2 * delayed(NBM, s.tau1) * survival / (s.theta2 + delayed(NPB, s.tau1))
        conversion = exp(-s.alpha2 * s.tau2 - s.r4 * delayed(APB, s.tau2))
        return [
            RateLaw(
                NBM,
                s.r1 * NBM * log((s.b1 + s.b2) / (s.theta1 + NBM)) - (s.m1 + s.m3) * NBM - release - s.c1 * NBM * ABM,
            ),
            RateLaw(NPB, arrival - s.c2 * NPB * APB - s.d1 * NPB),
            RateLaw(
                ABM,
                s.r3 * ABM * log(s.b3 / (s.theta3 + ABM))
                + s.m1 * NBM
                - s.m4 * ABM
                + s.m5 * conversion * delayed(AS, s.tau2),
            ),
            RateLaw(APB, s.m4 * ABM - s.d2 * APB),
            RateLaw(AS, s.r5 * AS * (1 - s.b4 * AS) - s.m5 * exp(-s.r4 * APB) * AS + s.m3 * NBM),
        ]


class LeukaemiaStemCellsElement(Element):
    """Normal and leukaemic stem cells that share a niche, with the progeny that feeds back on them.

    Stem cells renew symmetrically at the rate `r e^-psi1` and differentiate symmetrically at the rate
    `m e^-psi1 e^-psi3`, where `psi1 = beta1 (NS + AS)` counts the stem cells in the niche and
    `psi2 = beta2 (NPr + ND + AD)` and `psi3 = beta3 (NPr + ND + AD)` count their progeny. Asymmetric division at
    the rate `r e^-psi1 e^-psi2` and symmetric differentiation feed the progenitors and the leukaemic cells.

    Args:
        label:
            The label of the niche.

        NS:
            The initial normal stem cells.

        NPr:
            The initial normal progenitors.

        ND:
            The initial normal differentiated cells.

        AS:
            The initial leukaemic stem cells.

        AD:
            The initial differentiated leukaemic cells.

        r1:
            The symmetric renewal rate of normal stem cells.

        m1:
            The symmetric differentiation rate of normal stem cells.

        d1:
            The death rate of normal stem cells.

        r2:
            The asymmetric division rate of normal stem cells.

        r3:
            The proliferation rate of progenitors.

        d2:
            The rate at which progenitors leave their compartment.

        m2:
            The fraction of leaving progenitors that become differentiated cells.

        d3:
            The death rate of differentiated cells.

        r4:
            The symmetric renewal rate of leukaemic stem cells.

        m3:
            The symmetric differentiation rate of leukaemic stem cells.

        d4:
            The death rate of leukaemic stem cells.

        r5:
            The asymmetric division rate of leukaemic stem cells.

        d5:
            The death rate of differentiated leukaemic cells.

        beta1:
            The feedback gain of the stem cells.

        beta2:
            The feedback gain of the progeny on asymmetric division.

        beta3:
            The feedback gain of the progeny on differentiation.

    Examples:
        >>> from skcomplex.models.biology.oncology import LeukaemiaStemCellsElement
        >>> LeukaemiaStemCellsElement('niche').beta1
        0.01
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        NS: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        NPr: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ND: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        AS: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        m3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        d4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        r5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        d5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        beta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        beta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        beta3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.NS = NS
        self.NPr = NPr
        self.ND = ND
        self.AS = AS
        self.AD = AD
        self.r1 = r1
        self.m1 = m1
        self.d1 = d1
        self.r2 = r2
        self.r3 = r3
        self.d2 = d2
        self.m2 = m2
        self.d3 = d3
        self.r4 = r4
        self.m3 = m3
        self.d4 = d4
        self.r5 = r5
        self.d5 = d5
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regulated renewal, division, and differentiation of the normal and leukaemic lineages.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        NS, NPr, ND, AS, AD = s.NS, s.NPr, s.ND, s.AS, s.AD
        niche = exp(-s.beta1 * (NS + AS))
        progeny = NPr + ND + AD
        division, differentiation = exp(-s.beta2 * progeny), exp(-s.beta3 * progeny)
        return [
            RateLaw(NS, NS * niche * (s.r1 - s.m1 * differentiation) - s.d1 * NS),
            RateLaw(NPr, NS * niche * (s.r2 * division + 2 * s.m1 * differentiation) + (s.r3 - s.d2) * NPr),
            RateLaw(ND, s.m2 * s.d2 * NPr - s.d3 * ND),
            RateLaw(AS, AS * niche * (s.r4 - s.m3 * differentiation) - s.d4 * AS),
            RateLaw(AD, AS * niche * (s.r5 * division + 2 * s.m3 * differentiation) - s.d5 * AD),
        ]


@register_model(
    field='biology/oncology',
    tags=('stem cells', 'cancer stem cells', 'ode', 'feedback', 'random'),
    references=(BOOK, 'Calmelet et al. 2010'),
    defaults={'until': 2000.0, 'dt': 10.0},
)
def stem_cell_differentiation(
    *,
    m0: float = 0.005,
    beta0: float = 5.0,
    delta: float = 1.0,
    therapy: Literal['none', 'pulsed', 'continuous'] = 'continuous',
    start: float = 100.0,
    duration: float = 100.0,
    seed: int = 0,
) -> System:
    """Return the eight-compartment model of normal and malignant stem-cell lineages with feedback and therapy.

    Each lineage runs from quiescent stem cells through progenitors and basal cells to differentiated cells, and
    normal progenitors mutate into malignant ones. The mutation rate `m1 = m0 (1 + w)` carries the random factor `w`,
    which is drawn once per run from the uniform distribution on `[0, 1)` with the seed. The parameters and the
    initial cells are those of the book's table, with `beta0` at the top of its range. The book gives no value for
    `delta`, so it defaults to 1, and no times for the pulse of a pulsed therapy. The book prints the death rates
    off the pulse as the death rates times the dose, and the model keeps that. Time is in the units of the source.
    Under the continuous therapy the lineages settle on a steady state in which the mutation feeds a small malignant
    population.

    Args:
        m0:
            The mutation rate without the random factor.

        beta0:
            The feedback constant.

        delta:
            The weight of the renewal in the death the therapy causes to normal progenitors.

        therapy:
            The mode of therapy, `'none'`, `'pulsed'`, or `'continuous'`.

        start:
            The time the pulse of a pulsed therapy starts.

        duration:
            The duration of the pulse of a pulsed therapy.

        seed:
            The seed of the random factor of the mutation rate.

    Returns:
        system:
            The model, with the normal cells `tissue.NSq`, `tissue.NPr`, `tissue.NBs`, and `tissue.ND` and the
            malignant cells `tissue.ASq`, `tissue.APr`, `tissue.ABs`, and `tissue.AD`.

    Examples:
        >>> from skcomplex.models.biology.oncology import stem_cell_differentiation
        >>> 0.005 <= stem_cell_differentiation(seed=1).elements[0].m1 < 0.01
        True
    """
    m1 = m0 * (1 + np.random.default_rng(seed).uniform())
    tissue = StemCellDifferentiationElement(
        'tissue',
        m1=m1,
        beta0=beta0,
        delta=delta,
        therapy=therapy,
        start=start,
        duration=duration,
    )
    return System(elements=[tissue])


@register_model(
    field='biology/oncology',
    tags=('leukaemia', 'stem cells', 'delay', 'bone marrow'),
    references=(BOOK, 'Afenya, Ouifki, and Mundle 2019'),
    defaults={'until': 3000.0, 'dt': 10.0, 'solver': ScipyODESolver(method='BDF')},
)
def bone_marrow_blood_leukaemia(
    *,
    m1: float = 0.0013,
    m3: float = 1e-7,
    tau1: float = 96.0,
    tau2: float = 576.0,
) -> System:
    """Return the bone marrow and blood model of leukaemia, in which blasts crowd out the normal cells.

    Normal marrow cells renew, mutate into blasts and cancer stem cells, and mature into blood cells after the delay
    `tau1`. Blasts grow in the marrow, move to the blood, and inhibit the normal cells, and cancer stem cells convert
    into blasts after the delay `tau2`. Cells are counts, time is in hours, and the history before the start is the
    initial state. The book's table gives ranges, and the defaults are values in them: the largest growth rate of
    normal marrow cells and carrying capacity of blasts, the smallest delays, and the release rate `m2` that keeps
    the normal blood cells near their initial number. The book prints the arrival of marrow cells in the blood with
    a minus sign, and the model uses the plus sign of the release it balances. The blasts take over the marrow and
    the normal blood cells fall by orders of magnitude.

    Args:
        m1:
            The mutation rate of normal marrow cells into blasts, per hour.

        m3:
            The mutation rate of normal marrow cells into cancer stem cells, per hour.

        tau1:
            The maturation time of normal cells, in hours.

        tau2:
            The conversion time of cancer stem cells into blasts, in hours.

    Returns:
        system:
            The model, with the normal marrow and blood cells `patient.NBM` and `patient.NPB`, the blasts
            `patient.ABM` and `patient.APB`, and the cancer stem cells `patient.AS`.

    Examples:
        >>> from skcomplex.models.biology.oncology import bone_marrow_blood_leukaemia
        >>> bone_marrow_blood_leukaemia(tau1=144.0).elements[0].tau1
        144.0
    """
    return System(elements=[BoneMarrowBloodElement('patient', m1=m1, m3=m3, tau1=tau1, tau2=tau2)])


@register_model(
    field='biology/oncology',
    tags=('leukaemia', 'stem cells', 'ode', 'feedback', 'competition'),
    references=(BOOK, 'MacLean, Lo Celso, and Stumpf 2013'),
    defaults={'until': 500.0, 'dt': 1.0},
)
def leukaemia_stem_cells(
    NS: float = 100.0,
    AS: float = 1.0,
    *,
    r1: float = 0.6,
    m1: float = 0.2,
    d1: float = 0.1,
    r4: float = 0.7,
    m3: float = 0.2,
    d4: float = 0.1,
) -> System:
    """Return normal and leukaemic stem cells that compete for one niche through the feedback of their progeny.

    Both kinds of stem cells renew at the rate `r e^-psi1 - m e^-psi1 e^-psi3`, where `psi1` counts the stem cells
    in the niche and `psi3` their progeny, so the kind with the larger net renewal excludes the other. With the
    defaults the leukaemic stem cells win and take over, and with `r4 = 0.5` the leukaemia is controlled. The book
    gives ranges for the rates, and the defaults are values in them with the feedback gains `0.01` of the book.
    The book gives no initial cells, so a niche of 100 normal stem cells receives one leukaemic stem cell.

    Args:
        NS:
            The initial normal stem cells.

        AS:
            The initial leukaemic stem cells.

        r1:
            The symmetric renewal rate of normal stem cells.

        m1:
            The symmetric differentiation rate of normal stem cells.

        d1:
            The death rate of normal stem cells.

        r4:
            The symmetric renewal rate of leukaemic stem cells.

        m3:
            The symmetric differentiation rate of leukaemic stem cells.

        d4:
            The death rate of leukaemic stem cells.

    Returns:
        system:
            The model, with the normal cells `niche.NS`, `niche.NPr`, and `niche.ND` and the leukaemic cells
            `niche.AS` and `niche.AD`.

    Examples:
        >>> from skcomplex.models.biology.oncology import leukaemia_stem_cells
        >>> leukaemia_stem_cells(r4=0.5).elements[0].r4
        0.5
    """
    niche = LeukaemiaStemCellsElement('niche', NS=NS, AS=AS, r1=r1, m1=m1, d1=d1, r4=r4, m3=m3, d4=d4)
    return System(elements=[niche])
