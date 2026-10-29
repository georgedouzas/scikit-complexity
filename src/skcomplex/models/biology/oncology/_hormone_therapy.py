"""Oncology models: prostate and breast cancers under hormone therapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, log, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, CrossingLaw, RateLaw
from ....core.laws.utils import pulse_train
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ....errors import ValidationError
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'
PATIENT_FACTORS = {
    1: {
        'on': (0.956156, 0.00130716, 0.995718, 0.0, 0.0, 1.00181),
        'off': (1.00223, 0.1, 1.00392, 0.8),
        'cells': (17.84, 0.778, 0.382),
    },
    2: {
        'on': (0.925733, 0.00737385, 0.990242, 0.000130721, 0.0, 1.00339),
        'off': (1.00089, 0.1, 1.00147, 1.00079),
        'cells': (13.43, 1.57, 0.0),
    },
    3: {
        'on': (0.931568, 0.0, 0.992614, 0.00083635, 0.0, 1.00242),
        'off': (1.00306, 0.1, 1.0025, 1.00282),
        'cells': (11.02, 0.976, 0.0),
    },
}


def _hill(x: Expression, half: Expression, n: Expression) -> Expression:
    """Return the Hill function `x^n / (x^n + half^n)`."""
    return x**n / (x**n + half**n)


def _prostate_volume(s: SimpleNamespace) -> Expression:
    """Return the prostate volume in cm³ at the age of the patient, `v1 a^phi / (v2^phi + a^phi) + v3`."""
    age = s.age + s.t / 8760
    return s.v1 * age**s.phi / (s.v2**s.phi + age**s.phi) + s.v3


def _receptor_growth(s: SimpleNamespace, psi1: Expression, phi1: Expression) -> Expression:
    """Return the growth rate that the receptors activated by DHT and testosterone drive."""
    return psi1 * s.UD**2 / (s.psi2**2 + s.UD**2) + phi1 * s.UT**2 / (s.phi2**2 + s.UT**2)


def _receptor_death(s: SimpleNamespace, psi3: Expression) -> Expression:
    """Return the death rate that the activated receptors lower."""
    return psi3 / (s.psi4 + s.UD + s.UT)


def _quota_rate(s: SimpleNamespace, Q: Expression, q: Expression) -> Expression:
    """Return the uptake of androgen into a quota less its use for growth and its degradation."""
    uptake = s.vm * (s.qm - Q) / (s.qm - q) * s.U / (s.U + s.v50)
    return uptake - s.rm * (Q - q) - s.d3 * Q


class AndrogenSuppressionElement(Element):
    """Androgen-dependent and androgen-independent prostate cancer cells under continuous androgen suppression.

    The dependent cells follow `dAD/dt = r1 U (1 - b (AD + c1 AI)) AD - (d1 + m1)(1 - U / U0) AD`, the independent
    cells `dAI/dt = r2 (1 - b (AI + c2 AD)) AI + m1 (1 - U / U0) AD`, and the serum androgen
    `du/dt = -du (U - U0) - U0 du a`, which the drug efficacy `a` lowers to `U0 (1 - a)`.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial number of androgen-dependent cells.

        AI:
            The initial number of androgen-independent cells.

        U:
            The initial serum androgen concentration.

        r1:
            The growth rate of the dependent cells per unit of androgen.

        r2:
            The growth rate of the independent cells.

        b:
            The reciprocal carrying capacity.

        c1:
            The competition of the independent cells on the dependent ones.

        c2:
            The competition of the dependent cells on the independent ones.

        d1:
            The death rate of the dependent cells under androgen deprivation.

        m1:
            The rate at which androgen deprivation turns dependent cells independent.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        a:
            The efficacy of the suppression.

    Examples:
        >>> from skcomplex.models.biology.oncology import AndrogenSuppressionElement
        >>> AndrogenSuppressionElement('tumour', a=0.6).a
        0.6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.14,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.006,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.1e-11,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.064,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5e-05,
        U0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AI = AI
        self.U = U
        self.r1 = r1
        self.r2 = r2
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.d1 = d1
        self.m1 = m1
        self.U0 = U0
        self.du = du
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and transition of the cells and the suppression of the androgen.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        deprivation = 1 - s.U / s.U0
        return [
            RateLaw(s.AD, s.r1 * s.U * (1 - s.b * (s.AD + s.c1 * s.AI)) * s.AD - (s.d1 + s.m1) * deprivation * s.AD),
            RateLaw(s.AI, s.r2 * (1 - s.b * (s.AI + s.c2 * s.AD)) * s.AI + s.m1 * deprivation * s.AD),
            RateLaw(s.U, -s.du * (s.U - s.U0) - s.U0 * s.du * s.a),
        ]


class IntermittentSuppressionElement(Element):
    """Prostate cancer cells under androgen suppression that PSA thresholds switch on and off.

    The dependent cells follow `dAD/dt = r1 (k1 + (1 - k1) U / (U + k2)) AD - d1 (k3 + (1 - k3) U / (U + k4)) AD
    - m1 (1 - U / U0) AD`, the independent cells `dAI/dt = m1 (1 - U / U0) AD + r2 (1 - U / U0) AI - d2 AI`, and
    the serum androgen `du/dt = -du (U - U0) - U0 du u`. The therapy `u` turns 1 when the PSA `x1 AD + x2 AI` rises
    through `h1` and 0 when it falls through `h0`.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial number of androgen-dependent cells.

        AI:
            The initial number of androgen-independent cells.

        U:
            The initial serum androgen concentration.

        u:
            The initial therapy, 1 on and 0 off.

        r1:
            The growth rate of the dependent cells.

        r2:
            The growth rate of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        m1:
            The maximum rate at which dependent cells turn independent.

        k1:
            The growth of the dependent cells without androgen, relative to `r1`.

        k2:
            The androgen at which the growth of the dependent cells is half way.

        k3:
            The death of the dependent cells without androgen, relative to `d1`.

        k4:
            The androgen at which the death of the dependent cells is half way.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        x1:
            The PSA per dependent cell.

        x2:
            The PSA per independent cell.

        h0:
            The PSA below which the therapy stops.

        h1:
            The PSA above which the therapy starts.

    Examples:
        >>> from skcomplex.models.biology.oncology import IntermittentSuppressionElement
        >>> IntermittentSuppressionElement('tumour').h1
        15.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        u: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0204,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0242,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0076,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0168,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5e-05,
        k1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        k2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        k3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.0,
        k4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        U0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        x1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        x2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        h0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        h1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AI = AI
        self.U = U
        self.u = u
        self.r1 = r1
        self.r2 = r2
        self.d1 = d1
        self.d2 = d2
        self.m1 = m1
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.U0 = U0
        self.du = du
        self.x1 = x1
        self.x2 = x2
        self.h0 = h0
        self.h1 = h1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and transition of the cells, the androgen, and the switching of the therapy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        deprivation = 1 - s.U / s.U0
        growth = s.r1 * (s.k1 + (1 - s.k1) * s.U / (s.U + s.k2))
        death = s.d1 * (s.k3 + (1 - s.k3) * s.U / (s.U + s.k4))
        psa = s.x1 * s.AD + s.x2 * s.AI
        return [
            RateLaw(s.AD, (growth - death - s.m1 * deprivation) * s.AD),
            RateLaw(s.AI, s.m1 * deprivation * s.AD + (s.r2 * deprivation - s.d2) * s.AI),
            RateLaw(s.U, -s.du * (s.U - s.U0) - s.U0 * s.du * s.u),
            CrossingLaw(psa - s.h1, {s.u: 1}, direction=1, label='on'),
            CrossingLaw(psa - s.h0, {s.u: 0}, direction=-1, label='off'),
        ]


class CellQuotaProstateElement(Element):
    """Prostate cancer cells whose growth and mutation follow their androgen cell quota, with the PSA they release.

    The cells grow at the Droop rate `rm (1 - q_i / Q_i)`, die at `d_i`, and switch from dependent to independent at
    `mx1 m501^n / (Q1^n + m501^n)` and back at `mx2 Q2^n / (Q2^n + m502^n)`. The quotas follow
    `dQ_i/dt = vm (qm - Q_i) / (qm - q_i) U / (U + v50) - rm (Q_i - q_i) - d3 Q_i`, and the serum PSA
    `dP/dt = r0 (AD + AI) + r1 AD Q1^theta / (Q1^theta + rho1^theta) + r2 AI Q2^theta / (Q2^theta + rho2^theta)
    - du P`. The serum androgen relaxes at the rate `du1` to `Umin` during the `width` days of therapy that open
    every `period` days and to `U0` otherwise.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial number of androgen-dependent cells.

        AI:
            The initial number of androgen-independent cells.

        Q1:
            The initial cell quota of the dependent cells.

        Q2:
            The initial cell quota of the independent cells.

        P:
            The initial serum PSA.

        U:
            The initial serum androgen.

        rm:
            The maximum proliferation rate.

        q1:
            The minimum quota of the dependent cells.

        q2:
            The minimum quota of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        d3:
            The degradation rate of the quota.

        qm:
            The maximum quota.

        vm:
            The maximum uptake rate of androgen.

        v50:
            The serum androgen at which the uptake is half way.

        mx1:
            The maximum mutation rate from dependent to independent.

        mx2:
            The maximum mutation rate from independent to dependent.

        m501:
            The quota at which the mutation to independence is half way.

        m502:
            The quota at which the mutation back to dependence is half way.

        n:
            The steepness of the mutation rates.

        r0:
            The baseline PSA production per cell.

        r1:
            The androgen-driven PSA production per dependent cell.

        r2:
            The androgen-driven PSA production per independent cell.

        rho1:
            The quota at which the PSA production of the dependent cells is half way.

        rho2:
            The quota at which the PSA production of the independent cells is half way.

        theta:
            The steepness of the PSA production.

        du:
            The clearance rate of the PSA.

        U0:
            The normal serum androgen.

        Umin:
            The serum androgen under therapy.

        du1:
            The rate at which the serum androgen relaxes.

        period:
            The time between the starts of therapy periods.

        width:
            The duration of a therapy period.

    Examples:
        >>> from skcomplex.models.biology.oncology import CellQuotaProstateElement
        >>> CellQuotaProstateElement('tumour').qm
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Q1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        Q2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.4,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        rm: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        q1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        q2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.15,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.015,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.015,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.09,
        qm: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        vm: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.275,
        v50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        mx1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00015,
        mx2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        m501: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.08,
        m502: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.7,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        r0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.1,
        theta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        U0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        Umin: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        period: float = 500.0,
        width: float = 250.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AI = AI
        self.Q1 = Q1
        self.Q2 = Q2
        self.P = P
        self.U = U
        self.rm = rm
        self.q1 = q1
        self.q2 = q2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.qm = qm
        self.vm = vm
        self.v50 = v50
        self.mx1 = mx1
        self.mx2 = mx2
        self.m501 = m501
        self.m502 = m502
        self.n = n
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.rho1 = rho1
        self.rho2 = rho2
        self.theta = theta
        self.du = du
        self.U0 = U0
        self.Umin = Umin
        self.du1 = du1
        self.period = period
        self.width = width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and mutation of the cells, their quotas, the PSA, and the serum androgen.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        to_independent = s.mx1 * (1 - _hill(s.Q1, s.m501, s.n)) * s.AD
        to_dependent = s.mx2 * _hill(s.Q2, s.m502, s.n) * s.AI
        produced = s.r1 * s.AD * _hill(s.Q1, s.rho1, s.theta) + s.r2 * s.AI * _hill(s.Q2, s.rho2, s.theta)
        androgen = s.U0 - (s.U0 - s.Umin) * pulse_train(s.t, self.period, self.width)
        return [
            RateLaw(s.AD, s.rm * (1 - s.q1 / s.Q1) * s.AD - s.d1 * s.AD - to_independent + to_dependent),
            RateLaw(s.AI, s.rm * (1 - s.q2 / s.Q2) * s.AI - s.d2 * s.AI + to_independent - to_dependent),
            RateLaw(s.Q1, _quota_rate(s, s.Q1, s.q1)),
            RateLaw(s.Q2, _quota_rate(s, s.Q2, s.q2)),
            RateLaw(s.P, s.r0 * (s.AD + s.AI) + produced - s.du * s.P),
            RateLaw(s.U, s.du1 * (androgen - s.U)),
        ]


class AndrogenReceptorProstateElement(Element):
    """Normal epithelium and prostate cancer cells driven by activated androgen receptors, with tissue and serum PSA.

    The normal cells follow `dN/dt = r1 N (1 - (N + c1 (AD + AI)) / b1) - d1 N`, the dependent cells
    `dAD/dt = r2 AD (1 - m1)(1 - L) - d2 AD`, and the independent cells `dAI/dt = r3 AI (1 - (c1 N + AD + AI) / b3)
    - d3 AI + m1 r2 AD (1 - L)`, where `L = (c1 N + AD + AI) / b2`. The rates are
    `r_i = psi1_i UD^2 / (psi2^2 + UD^2) + phi1_i UT^2 / (phi2^2 + UT^2)` and `d_i = psi3_i / (psi4 + UD + UT)` for
    the receptors activated by DHT, `UD`, and by testosterone, `UT`. The tissue PSA follows
    `dUIP/dt = -du1 UIP Nf - gamma_p UIP (AD + AI)^2 / (KP + AD + AI) - du2 UIP + r4`, with the production
    `r4 = sum_i beta_i (UT + UD + sigma) X_i` over the cell types, and the serum PSA gains the leaks and is cleared
    at `du3`. The capacity of the normal cells `b1` and their share `Nf = N / b1` follow the prostate volume
    `v1 a^phi / (v2^phi + a^phi) + v3` at the age `a`, divided by the volume `Vc` of a billion cells.

    Args:
        label:
            The label of the prostate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial normal epithelial cells.

        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        UIP:
            The initial tissue PSA.

        UP:
            The initial serum PSA.

        UT:
            The concentration of receptors activated by testosterone.

        UD:
            The concentration of receptors activated by dihydrotestosterone.

        m1:
            The fraction of the divisions of dependent cells that yields independent cells.

        b2:
            The carrying capacity of the dependent cells.

        b3:
            The carrying capacity of the independent cells.

        c1:
            The interaction between normal and cancer cells.

        psi1_n:
            The maximum growth of the normal cells driven by DHT.

        psi1_d:
            The maximum growth of the dependent cells driven by DHT.

        psi1_i:
            The maximum growth of the independent cells driven by DHT.

        phi1_n:
            The maximum growth of the normal cells driven by testosterone.

        phi1_d:
            The maximum growth of the dependent cells driven by testosterone.

        phi1_i:
            The maximum growth of the independent cells driven by testosterone.

        psi2:
            The DHT-activated receptors at which the growth is half way.

        phi2:
            The testosterone-activated receptors at which the growth is half way.

        psi3_n:
            The death of the normal cells.

        psi3_d:
            The death of the dependent cells.

        psi3_i:
            The death of the independent cells.

        psi4:
            The activated receptors at which the death is halved.

        Vc:
            The volume of a billion cells.

        v1:
            The prostate volume that ageing adds.

        v2:
            The age at which half of that volume is added.

        v3:
            The prostate volume of a young man.

        phi:
            The steepness of the growth of the prostate with age.

        age:
            The age of the patient at the start.

        du1:
            The rate at which tissue PSA leaks through normal epithelium.

        du2:
            The decay rate of tissue PSA.

        du3:
            The clearance rate of serum PSA.

        gamma_p:
            The rate at which tissue PSA leaks through cancer cells.

        KP:
            The cancer cells at which the leak per cell is half its maximum.

        beta_n:
            The PSA production of the normal cells.

        beta_d:
            The PSA production of the dependent cells.

        beta_i:
            The PSA production of the independent cells.

        sigma:
            The PSA production independent of the receptors.

    Examples:
        >>> from skcomplex.models.biology.oncology import AndrogenReceptorProstateElement
        >>> AndrogenReceptorProstateElement('prostate').KP
        800.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        UIP: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        UP: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        UT: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        UD: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 22.4,
        b3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 22.4,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        psi1_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001202,
        psi1_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001048,
        psi1_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.8e-05,
        phi1_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001281,
        phi1_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00111,
        phi1_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.000279,
        psi2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.355,
        phi2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 49.587,
        psi3_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.008,
        psi3_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        psi3_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        psi4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 62.162,
        Vc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.56,
        v1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 18.397,
        v2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 63.753,
        v3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 28.4,
        phi: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.0,
        age: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 60.0,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001515,
        du2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05635,
        du3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.776,
        gamma_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3125,
        KP: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 800.0,
        beta_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0022622,
        beta_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        beta_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.003,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.AD = AD
        self.AI = AI
        self.UIP = UIP
        self.UP = UP
        self.UT = UT
        self.UD = UD
        self.m1 = m1
        self.b2 = b2
        self.b3 = b3
        self.c1 = c1
        self.psi1_n = psi1_n
        self.psi1_d = psi1_d
        self.psi1_i = psi1_i
        self.phi1_n = phi1_n
        self.phi1_d = phi1_d
        self.phi1_i = phi1_i
        self.psi2 = psi2
        self.phi2 = phi2
        self.psi3_n = psi3_n
        self.psi3_d = psi3_d
        self.psi3_i = psi3_i
        self.psi4 = psi4
        self.Vc = Vc
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.phi = phi
        self.age = age
        self.du1 = du1
        self.du2 = du2
        self.du3 = du3
        self.gamma_p = gamma_p
        self.KP = KP
        self.beta_n = beta_n
        self.beta_d = beta_d
        self.beta_i = beta_i
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth of the normal and cancer cells and the balance of tissue and serum PSA.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        b1 = _prostate_volume(s) / s.Vc
        cancer = s.AD + s.AI
        crowding = (s.c1 * s.N + cancer) / s.b2
        division = _receptor_growth(s, s.psi1_d, s.phi1_d) * s.AD * (1 - crowding)
        normal_leak = s.du1 * s.UIP * s.N / b1
        cancer_leak = s.gamma_p * s.UIP * cancer**2 / (s.KP + cancer)
        activation = s.UT + s.UD + s.sigma
        production = activation * (s.beta_n * s.N + s.beta_d * s.AD + s.beta_i * s.AI)
        return [
            RateLaw(
                s.N,
                _receptor_growth(s, s.psi1_n, s.phi1_n) * s.N * (1 - (s.N + s.c1 * cancer) / b1)
                - _receptor_death(s, s.psi3_n) * s.N,
            ),
            RateLaw(s.AD, (1 - s.m1) * division - _receptor_death(s, s.psi3_d) * s.AD),
            RateLaw(
                s.AI,
                _receptor_growth(s, s.psi1_i, s.phi1_i) * s.AI * (1 - (s.c1 * s.N + cancer) / s.b3)
                - _receptor_death(s, s.psi3_i) * s.AI
                + s.m1 * division,
            ),
            RateLaw(s.UIP, production - normal_leak - cancer_leak - s.du2 * s.UIP),
            RateLaw(s.UP, normal_leak + cancer_leak - s.du3 * s.UP),
        ]


class PiecewiseLinearProstateElement(Element):
    """Androgen-dependent, reversibly and irreversibly independent prostate cancer cells under intermittent therapy.

    During therapy, `u = 1`, the cells follow `dAD/dt = w11_on AD`, `dAIR/dt = w21_on AD + w22_on AIR`, and
    `dAII/dt = w31_on AD + w32_on AIR + w33_on AII`. Off therapy they follow `dAD/dt = w11_off AD + w12_off AIR`,
    `dAIR/dt = w22_off AIR`, and `dAII/dt = w33_off AII`. The therapy turns on when the PSA `x1 AD + x2 AIR + x3 AII`
    rises through `h1` and off when it falls through `h0`.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial androgen-dependent cells.

        AIR:
            The initial reversibly independent cells.

        AII:
            The initial irreversibly independent cells.

        u:
            The initial therapy, 1 on and 0 off.

        w11_on:
            The net growth rate of the dependent cells during therapy.

        w21_on:
            The rate at which dependent cells adapt during therapy.

        w22_on:
            The net growth rate of the adapted cells during therapy.

        w31_on:
            The rate at which dependent cells mutate during therapy.

        w32_on:
            The rate at which adapted cells mutate during therapy.

        w33_on:
            The net growth rate of the mutated cells during therapy.

        w11_off:
            The net growth rate of the dependent cells off therapy.

        w12_off:
            The rate at which adapted cells return to dependence off therapy.

        w22_off:
            The net growth rate of the adapted cells off therapy.

        w33_off:
            The net growth rate of the mutated cells off therapy.

        x1:
            The PSA per dependent cell.

        x2:
            The PSA per adapted cell.

        x3:
            The PSA per mutated cell.

        h0:
            The PSA below which the therapy stops.

        h1:
            The PSA above which the therapy starts.

    Examples:
        >>> from skcomplex.models.biology.oncology import PiecewiseLinearProstateElement
        >>> PiecewiseLinearProstateElement('tumour').h0
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 17.84,
        AIR: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.778,
        AII: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.382,
        u: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        w11_on: Annotated[float, ParameterSpec()] = -0.043844,
        w21_on: Annotated[float, ParameterSpec()] = 0.00130716,
        w22_on: Annotated[float, ParameterSpec()] = -0.004282,
        w31_on: Annotated[float, ParameterSpec()] = 0.0,
        w32_on: Annotated[float, ParameterSpec()] = 0.0,
        w33_on: Annotated[float, ParameterSpec()] = 0.00181,
        w11_off: Annotated[float, ParameterSpec()] = 0.00223,
        w12_off: Annotated[float, ParameterSpec()] = 0.1,
        w22_off: Annotated[float, ParameterSpec()] = 0.00392,
        w33_off: Annotated[float, ParameterSpec()] = -0.2,
        x1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        x2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        x3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        h0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        h1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AIR = AIR
        self.AII = AII
        self.u = u
        self.w11_on = w11_on
        self.w21_on = w21_on
        self.w22_on = w22_on
        self.w31_on = w31_on
        self.w32_on = w32_on
        self.w33_on = w33_on
        self.w11_off = w11_off
        self.w12_off = w12_off
        self.w22_off = w22_off
        self.w33_off = w33_off
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.h0 = h0
        self.h1 = h1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the linear growth and transitions of each mode and the switching of the therapy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        on, off = s.u, 1 - s.u
        psa = s.x1 * s.AD + s.x2 * s.AIR + s.x3 * s.AII
        return [
            RateLaw(s.AD, on * s.w11_on * s.AD + off * (s.w11_off * s.AD + s.w12_off * s.AIR)),
            RateLaw(s.AIR, on * (s.w21_on * s.AD + s.w22_on * s.AIR) + off * s.w22_off * s.AIR),
            RateLaw(s.AII, on * (s.w31_on * s.AD + s.w32_on * s.AIR + s.w33_on * s.AII) + off * s.w33_off * s.AII),
            CrossingLaw(psa - s.h1, {s.u: 1}, direction=1, label='on'),
            CrossingLaw(psa - s.h0, {s.u: 0}, direction=-1, label='off'),
        ]


class TamoxifenTumourElement(Element):
    """Breast tumour in three Gompertzian cell-cycle phases under tamoxifen, its metabolite, and the lymphocytes.

    The phases follow `dAg/dt = -mg Ag L + 2 mm Am L - a1 Ag (UX + delta1 UH)`, `dAs/dt = -ms As + mg Ag L`, and
    `dAm/dt = -mm Am L + ms As`, with `L = log(Asat / (Ag + As + Am))`. Tamoxifen passes from the gut `X0` through
    `X1` to the plasma `X2`, `dX0/dt = -k01 X0 + u(t)`, `dX1/dt = k01 X0 - k12 X1`,
    `dX2/dt = k12 X1 - (du1 + k23) X2`, and the plasma turns it into 4-hydroxytamoxifen,
    `dX3/dt = k23 X2 - du2 X3`, with the concentrations `UX = X2 / V` and `UH = X3 / V`. The lymphocytes follow
    `dE/dt = s - d1 E - a2 E (UX + delta2 UH)`, and tamoxifen enters the gut at the constant rate `u`.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Ag:
            The initial volume of cells in the growth phase.

        As:
            The initial volume of cells in the synthesis phase.

        Am:
            The initial volume of cells in mitosis.

        X0:
            The initial tamoxifen in the gut.

        X1:
            The initial tamoxifen in the second compartment.

        X2:
            The initial tamoxifen in the plasma.

        X3:
            The initial 4-hydroxytamoxifen.

        E:
            The initial circulating lymphocytes, in millions.

        mg:
            The transit rate out of the growth phase.

        ms:
            The transit rate out of the synthesis phase.

        mm:
            The transit rate out of mitosis.

        Asat:
            The volume at which growth saturates.

        a1:
            The kill rate of the tumour per unit of drug.

        a2:
            The kill rate of the lymphocytes per unit of drug.

        delta1:
            The tumour kill of the metabolite relative to tamoxifen.

        delta2:
            The lymphocyte kill of the metabolite relative to tamoxifen.

        k01:
            The transfer rate from the gut.

        k12:
            The transfer rate to the plasma.

        k23:
            The rate at which the plasma metabolizes tamoxifen.

        du1:
            The elimination rate of tamoxifen.

        du2:
            The elimination rate of the metabolite.

        V:
            The volume of distribution.

        s:
            The influx of lymphocytes, in millions per unit of time.

        d1:
            The death rate of lymphocytes.

        u:
            The infusion rate of tamoxifen.

    Examples:
        >>> from skcomplex.models.biology.oncology import TamoxifenTumourElement
        >>> TamoxifenTumourElement('tumour').Asat
        10000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Ag: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 900.0,
        As: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        Am: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        X0: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        X1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        X2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        X3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        mg: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0013,
        ms: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.039,
        mm: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0169,
        Asat: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10000.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0062,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        delta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 25.0,
        delta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 25.0,
        k01: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.048,
        k12: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.993,
        k23: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 35.932,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.145,
        du2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 39.525,
        V: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.592,
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.121,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.012,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 12.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Ag = Ag
        self.As = As
        self.Am = Am
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.E = E
        self.mg = mg
        self.ms = ms
        self.mm = mm
        self.Asat = Asat
        self.a1 = a1
        self.a2 = a2
        self.delta1 = delta1
        self.delta2 = delta2
        self.k01 = k01
        self.k12 = k12
        self.k23 = k23
        self.du1 = du1
        self.du2 = du2
        self.V = V
        self.s = s
        self.d1 = d1
        self.u = u

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the cell cycle under the drug, the pharmacokinetics, and the lymphocytes.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        room = log(s.Asat / (s.Ag + s.As + s.Am))
        UX, UH = s.X2 / s.V, s.X3 / s.V
        return [
            RateLaw(s.Ag, -s.mg * s.Ag * room + 2 * s.mm * s.Am * room - s.a1 * s.Ag * (UX + s.delta1 * UH)),
            RateLaw(s.As, -s.ms * s.As + s.mg * s.Ag * room),
            RateLaw(s.Am, -s.mm * s.Am * room + s.ms * s.As),
            RateLaw(s.X0, -s.k01 * s.X0 + s.u),
            RateLaw(s.X1, s.k01 * s.X0 - s.k12 * s.X1),
            RateLaw(s.X2, s.k12 * s.X1 - (s.du1 + s.k23) * s.X2),
            RateLaw(s.X3, s.k23 * s.X2 - s.du2 * s.X3),
            RateLaw(s.E, s.s - s.d1 * s.E - s.a2 * s.E * (UX + s.delta2 * UH)),
        ]


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'androgen'),
    references=(BOOK, 'Zazoua and Wang 2019'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def continuous_androgen_suppression(
    AD: float = 10.0,
    AI: float = 0.14,
    U: float = 20.0,
    *,
    r1: float = 0.025,
    r2: float = 0.006,
    b: float = 0.091e-9,
    c1: float = 0.9,
    c2: float = 0.8,
    d1: float = 0.064,
    m1: float = 5e-5,
    U0: float = 20.0,
    du: float = 0.08,
    a: float = 0.9,
) -> System:
    """Return prostate cancer under continuous androgen suppression, which androgen-independent cells survive.

    The suppression lowers the serum androgen from `U0` to `U0 (1 - a)` at the rate `du`. The androgen-dependent cells
    grow in proportion to the androgen and die and turn independent in proportion to the deprivation `1 - U / U0`, so
    they decline once the suppression is deep enough, while the independent cells grow regardless and cause the
    relapse. Cells are counted, androgen is in ng/mL, and time is in days. The book's efficacy lies between 0.6 and
    0.9, and only the upper end makes the dependent cells decline.

    Args:
        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        U:
            The initial serum androgen.

        r1:
            The growth rate of the dependent cells per unit of androgen.

        r2:
            The growth rate of the independent cells.

        b:
            The reciprocal carrying capacity.

        c1:
            The competition of the independent cells on the dependent ones.

        c2:
            The competition of the dependent cells on the independent ones.

        d1:
            The death rate of the dependent cells under androgen deprivation.

        m1:
            The rate at which androgen deprivation turns dependent cells independent.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        a:
            The efficacy of the suppression.

    Returns:
        system:
            The model, with the cells `tumour.AD` and `tumour.AI` and the androgen `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import continuous_androgen_suppression
        >>> continuous_androgen_suppression(a=0.6).elements[0].a
        0.6
    """
    tumour = AndrogenSuppressionElement(
        'tumour',
        AD=AD,
        AI=AI,
        U=U,
        r1=r1,
        r2=r2,
        b=b,
        c1=c1,
        c2=c2,
        d1=d1,
        m1=m1,
        U0=U0,
        du=du,
        a=a,
    )
    return System(elements=[tumour])


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'events', 'intermittent'),
    references=(BOOK, 'Ideta et al. 2008'),
    defaults={'until': 600.0, 'dt': 2.0},
)
def ideta_intermittent_androgen(
    AD: float = 5.0,
    AI: float = 0.0,
    *,
    r1: float = 0.0204,
    r2: float = 0.0242,
    d1: float = 0.0076,
    d2: float = 0.0168,
    m1: float = 5e-5,
    k1: float = 0.0,
    k2: float = 2.0,
    k3: float = 8.0,
    k4: float = 0.5,
    U0: float = 30.0,
    du: float = 0.08,
    h0: float = 10.0,
    h1: float = 15.0,
) -> System:
    """Return the Ideta model of intermittent androgen suppression, switched on and off by PSA thresholds.

    The therapy starts when the serum PSA, the number of cancer cells, rises through `h1` and stops when it falls
    through `h0`. The androgen follows with the delay `1 / du`, so the dependent cells shrink during therapy and regrow
    after it, and the PSA cycles between the thresholds while the independent cells stay few. Cells are counted, the
    PSA is in ng/mL, the androgen in nmol/L, starting at its normal level `U0` within the book's 8 to 35, and time is
    in days.

    Args:
        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        r1:
            The growth rate of the dependent cells.

        r2:
            The growth rate of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        m1:
            The maximum rate at which dependent cells turn independent.

        k1:
            The growth of the dependent cells without androgen, relative to `r1`.

        k2:
            The androgen at which the growth of the dependent cells is half way.

        k3:
            The death of the dependent cells without androgen, relative to `d1`.

        k4:
            The androgen at which the death of the dependent cells is half way.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        h0:
            The PSA below which the therapy stops.

        h1:
            The PSA above which the therapy starts.

    Returns:
        system:
            The model, with the cells `tumour.AD` and `tumour.AI`, the androgen `tumour.U`, the therapy `tumour.u`, and
            the switches as `on` and `off` events.

    Examples:
        >>> from skcomplex.models.biology.oncology import ideta_intermittent_androgen
        >>> ideta_intermittent_androgen(h1=20.0).elements[0].h1
        20.0
    """
    tumour = IntermittentSuppressionElement(
        'tumour',
        AD=AD,
        AI=AI,
        U=U0,
        r1=r1,
        r2=r2,
        d1=d1,
        d2=d2,
        m1=m1,
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        U0=U0,
        du=du,
        h0=h0,
        h1=h1,
    )
    return System(elements=[tumour])


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'cell quota', 'intermittent'),
    references=(BOOK, 'Portz, Kuang, and Nagy 2012'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def prostate_cell_quota(
    AD: float = 15.0,
    AI: float = 0.0,
    *,
    rm: float = 0.025,
    q1: float = 0.3,
    q2: float = 0.15,
    d1: float = 0.015,
    d2: float = 0.015,
    r0: float = 0.01,
    r1: float = 0.15,
    r2: float = 0.15,
    n: float = 3.0,
    theta: float = 3.0,
    U0: float = 15.0,
    Umin: float = 0.5,
    du1: float = 0.08,
    period: float = 500.0,
    width: float = 250.0,
) -> System:
    """Return the Portz cell-quota model of prostate cancer, whose cells grow on their internal androgen.

    The androgen-dependent and independent cells grow at the Droop rate `rm (1 - q_i / Q_i)` of their cell quotas,
    which take up serum androgen, and the PSA they release follows the quotas. A deprived dependent cell turns
    independent faster, and an independent cell full of androgen turns back. The serum androgen follows the
    exponential relaxation of the book's equation 7.18 between the normal level `U0` and `Umin`, with therapy during
    the first `width` days of every `period`. The book gives neither the steepness `n` of the mutation rates nor
    `theta` of the PSA production, nor the serum levels, their relaxation rate, and the schedule, so these defaults
    are chosen, and the other defaults lie within the book's ranges. Cells are in billions, quotas in nM, PSA in
    ng/mL, and time in days. The PSA starts near its value for the initial quotas.

    Args:
        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        rm:
            The maximum proliferation rate.

        q1:
            The minimum quota of the dependent cells.

        q2:
            The minimum quota of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        r0:
            The baseline PSA production per cell.

        r1:
            The androgen-driven PSA production per dependent cell.

        r2:
            The androgen-driven PSA production per independent cell.

        n:
            The steepness of the mutation rates.

        theta:
            The steepness of the PSA production.

        U0:
            The normal serum androgen.

        Umin:
            The serum androgen under therapy.

        du1:
            The rate at which the serum androgen relaxes.

        period:
            The time between the starts of therapy periods.

        width:
            The duration of a therapy period.

    Returns:
        system:
            The model, with the cells `tumour.AD` and `tumour.AI`, the quotas `tumour.Q1` and `tumour.Q2`, the PSA
            `tumour.P`, and the androgen `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import prostate_cell_quota
        >>> prostate_cell_quota(q1=0.45).elements[0].q1
        0.45
    """
    tumour = CellQuotaProstateElement(
        'tumour',
        AD=AD,
        AI=AI,
        U=U0,
        rm=rm,
        q1=q1,
        q2=q2,
        d1=d1,
        d2=d2,
        r0=r0,
        r1=r1,
        r2=r2,
        n=n,
        theta=theta,
        U0=U0,
        Umin=Umin,
        du1=du1,
        period=period,
        width=width,
    )
    return System(elements=[tumour])


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'androgen receptor', 'PSA'),
    references=(BOOK, 'Jain et al. 2011'),
    defaults={'until': 26280.0, 'dt': 168.0, 'solver': ScipyODESolver(method='LSODA')},
)
def prostate_androgen_receptor(
    N: float = 1.5,
    AD: float = 1.0,
    AI: float = 0.0,
    *,
    UT: float = 40.0,
    UD: float = 40.0,
    m1: float = 1e-4,
    age: float = 60.0,
) -> System:
    """Return the Jain model of prostate epithelium and cancer driven by activated androgen receptors, with PSA.

    The growth and death rates of the normal, androgen-dependent, and androgen-independent cells follow the receptors
    activated by testosterone, `UT`, and dihydrotestosterone, `UD`, which are parameters, so androgen ablation is a
    lower activation. The normal cells fill a capacity that grows with the age-dependent prostate volume, and the PSA
    the cells produce in the tissue leaks into the serum through the normal epithelium and, faster, through the
    tumour. Cells are in billions, receptors in nM, PSA in ng/mL, and time in hours, the age in years. The book
    leaves several parts open, filled as follows. The half saturations `psi2`, `phi2`, and `psi4` of the cancer cells
    are those of the normal cells. The death rate is `psi3_i / (psi4 + UD + UT)`. One activation `UT`, `UD` holds
    for all cell types, and so does `sigma`. The independent cells gain the part `m1` of the divisions of the
    dependent cells. The negligible interaction `c1` is 0. The capacity of the normal cells is the prostate volume
    over the volume of a billion cells, and the PSA leak rate per million cells becomes `0.3125` per billion.

    Args:
        N:
            The initial normal epithelial cells.

        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        UT:
            The concentration of receptors activated by testosterone.

        UD:
            The concentration of receptors activated by dihydrotestosterone.

        m1:
            The fraction of the divisions of dependent cells that yields independent cells.

        age:
            The age of the patient at the start.

    Returns:
        system:
            The model, with the cells `prostate.N`, `prostate.AD`, and `prostate.AI`, and the PSA `prostate.UIP` in
            the tissue and `prostate.UP` in the serum.

    Examples:
        >>> from skcomplex.models.biology.oncology import prostate_androgen_receptor
        >>> prostate_androgen_receptor(UT=5.0).elements[0].UT
        5.0
    """
    prostate = AndrogenReceptorProstateElement('prostate', N=N, AD=AD, AI=AI, UT=UT, UD=UD, m1=m1, age=age)
    return System(elements=[prostate])


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'prostate cancer', 'piecewise linear', 'events', 'intermittent'),
    references=(BOOK, 'Hirata, Bruchovsky, and Aihara 2010'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def piecewise_linear_ias(patient: int = 1, *, h0: float = 4.0, h1: float = 10.0) -> System:
    """Return the Hirata piecewise-linear model of intermittent androgen suppression for one of three patient types.

    The androgen-dependent, adapted, and mutated cells grow and change into each other linearly, with one set of
    rates during therapy and one off therapy, and the therapy switches on when the PSA, the sum of the cells, rises
    through `h1` and off when it falls through `h0`. The rates are the book's daily factors, `w = omega - 1` on the
    diagonal and `w = omega` off it. Patient 1 does not relapse, patient 2 relapses late, and patient 3 relapses,
    once the mutated cells outgrow the therapy. The book only states that therapy starts above 10 ng/mL, so the
    therapy stops below 4 ng/mL and is on at the start when the PSA exceeds `h1`. The cells are in units of the PSA,
    ng/mL, and time is in days.

    Args:
        patient:
            The patient type, 1, 2, or 3, which sets the rates and the initial cells.

        h0:
            The PSA below which the therapy stops.

        h1:
            The PSA above which the therapy starts.

    Returns:
        system:
            The model, with the cells `tumour.AD`, `tumour.AIR`, and `tumour.AII`, the therapy `tumour.u`, and the
            switches as `on` and `off` events.

    Raises:
        ValidationError:
            The patient type is not 1, 2, or 3.

    Examples:
        >>> from skcomplex.models.biology.oncology import piecewise_linear_ias
        >>> piecewise_linear_ias(patient=3).elements[0].AD
        11.02
    """
    if patient not in PATIENT_FACTORS:
        error_msg = f'The patient type must be 1, 2, or 3, got {patient!r}.'
        raise ValidationError(error_msg)
    factors = PATIENT_FACTORS[patient]
    on11, on21, on22, on31, on32, on33 = factors['on']
    off11, off12, off22, off33 = factors['off']
    AD, AIR, AII = factors['cells']
    tumour = PiecewiseLinearProstateElement(
        'tumour',
        AD=AD,
        AIR=AIR,
        AII=AII,
        u=float(h1 < AD + AIR + AII),
        w11_on=on11 - 1,
        w21_on=on21,
        w22_on=on22 - 1,
        w31_on=on31,
        w32_on=on32,
        w33_on=on33 - 1,
        w11_off=off11 - 1,
        w12_off=off12,
        w22_off=off22 - 1,
        w33_off=off33 - 1,
        h0=h0,
        h1=h1,
    )
    return System(elements=[tumour])


@register_model(
    field='biology/oncology',
    tags=('hormone therapy', 'ode', 'breast cancer', 'cell cycle', 'pharmacokinetics'),
    references=(BOOK, 'Chen, Kirkby, and Jena 2012'),
    defaults={'until': 2400.0, 'dt': 8.0, 'solver': ScipyODESolver(method='LSODA')},
)
def tamoxifen_cell_cycle(
    Ag: float = 900.0,
    As: float = 50.0,
    Am: float = 50.0,
    *,
    u: float = 12.5,
    a1: float = 0.0062,
    a2: float = 0.010,
) -> System:
    """Return a breast tumour in Gompertzian cell-cycle phases under daily tamoxifen and its active metabolite.

    The cells pass from the growth phase through synthesis to mitosis and divide back into the growth phase, at
    rates that slow as the tumour approaches `Asat`. Tamoxifen is absorbed from the gut into the plasma, where it is
    metabolized into 4-hydroxytamoxifen, and both kill cells in the growth phase, the metabolite 25 times more. They
    also kill circulating lymphocytes, which stay at `s / d1` without the drug. The book gives no infusion rate, so the
    default `u` is 300 μg a day, given continuously. Volumes are in mm³, drug masses in μg, concentrations in
    μg/mL, lymphocytes in millions, and time in hours.

    Args:
        Ag:
            The initial volume of cells in the growth phase.

        As:
            The initial volume of cells in the synthesis phase.

        Am:
            The initial volume of cells in mitosis.

        u:
            The infusion rate of tamoxifen.

        a1:
            The kill rate of the tumour per unit of drug.

        a2:
            The kill rate of the lymphocytes per unit of drug.

    Returns:
        system:
            The model, with the phases `tumour.Ag`, `tumour.As`, and `tumour.Am`, the drug masses `tumour.X0` to
            `tumour.X3`, and the lymphocytes `tumour.E`.

    Examples:
        >>> from skcomplex.models.biology.oncology import tamoxifen_cell_cycle
        >>> tamoxifen_cell_cycle(u=0.0).elements[0].u
        0.0
    """
    tumour = TamoxifenTumourElement('tumour', Ag=Ag, As=As, Am=Am, u=u, a1=a1, a2=a2)
    return System(elements=[tumour])
