"""Oncology models: tumour-immune interactions under adoptive therapy, vaccines, cytokines, and checkpoints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import log

from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rate, Symbols, delayed, pulse_train, step, where
from ....core.runtime import register_model
from ....core.solvers import ScipyODE
from ....core.system import System

BOOK_REFERENCE = 'Padmanabhan, Meskin, and Al Moustafa 2021'
STIFF_SOLVER = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)


def _schedule(t: Expression, period: float, width: float, start: float, doses: int) -> Expression:
    """Return a train of `doses` unit pulses of a width every period from a start."""
    return pulse_train(t, period, width, start=start) * (1 - step(t, start + doses * period))


def _saturation(x: Expression, half: Expression) -> Expression:
    """Return the Michaelis-Menten fraction `x / (half + x)`."""
    return x / (half + x)


class KirschnerPanettaTumour(Element):
    """Effector cells, tumour cells, and IL-2 under adoptive cellular therapy and IL-2 injection.

    The effector cells follow `dE/dt = alpha_a A - d1 E + a1 u1 + rho1 E C / (alpha1 + C)`, the tumour
    `dA/dt = r1 A (1 - b A) - c1 E A / (alpha2 + A)`, and the IL-2 `dC/dt = a2 u2 + rho2 E A / (alpha3 + A) - du C`,
    with constant inputs `u1` and `u2`.

    Args:
        label:
            The label of the tumour site.

        E:
            The initial effector cells.

        A:
            The initial tumour cells.

        C:
            The initial IL-2 concentration.

        alpha_a:
            The antigenicity of the tumour.

        d1:
            The death rate of effector cells.

        rho1:
            The proliferation rate of effector cells stimulated by IL-2.

        alpha1:
            The IL-2 at half the proliferation.

        r1:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

        alpha2:
            The tumour at half the kill.

        rho2:
            The IL-2 production rate by effector-tumour encounters.

        alpha3:
            The tumour at half the IL-2 production.

        du:
            The loss rate of IL-2.

        a1:
            The effector cells per unit of adoptive therapy.

        a2:
            The IL-2 per unit of IL-2 injection.

        u1:
            The adoptive therapy input.

        u2:
            The IL-2 injection input.

    Examples:
        >>> from skcomplex.models.biology.oncology import KirschnerPanettaTumour
        >>> KirschnerPanettaTumour('site').rho1
        0.1245
    """

    E: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    A: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    C: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    alpha_a: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.03, domain=Domain.NONNEGATIVE)
    rho1: float = parameter(default=0.1245, domain=Domain.NONNEGATIVE)
    alpha1: float = parameter(default=2e7, domain=Domain.POSITIVE)
    r1: float = parameter(default=0.18, domain=Domain.POSITIVE)
    b: float = parameter(default=1e-9, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    alpha2: float = parameter(default=1e5, domain=Domain.POSITIVE)
    rho2: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    alpha3: float = parameter(default=1e3, domain=Domain.POSITIVE)
    du: float = parameter(default=10.0, domain=Domain.POSITIVE)
    a1: float = parameter(default=500.0, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=7e7, domain=Domain.NONNEGATIVE)
    u1: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    u2: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of effector cells, tumour cells, and IL-2.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, A, C = s.E, s.A, s.C
        return [
            Rate(E, s.alpha_a * A - s.d1 * E + s.a1 * s.u1 + s.rho1 * E * _saturation(C, s.alpha1)),
            Rate(A, s.r1 * A * (1 - s.b * A) - s.c1 * E * _saturation(A, s.alpha2)),
            Rate(C, s.a2 * s.u2 + s.rho2 * E * _saturation(A, s.alpha3) - s.du * C),
        ]


class VaccinatedTumour(Element):
    """Helper T cells, cytotoxic T cells, tumour cells, dendritic cells, and IL-2 under dendritic-cell vaccination.

    The helper cells follow `dEH/dt = s1 + rho1 ED r1 EH (1 - b1 EH) - d1 EH`, the cytotoxic cells
    `dE/dt = s2 + rho2 C (A + ED) r2 E (1 - b2 E) - d2 E`, the tumour `dA/dt = r3 A (1 - b3 A) - c1 A E`, the
    dendritic cells `dED/dt = -c2 E ED + u(t)`, and the IL-2 `dC/dt = r4 EH ED - c3 E C - d3 C`. The vaccine `u` is
    `dose` for `width` time units every `period` from `start`, `doses` times.

    Args:
        label:
            The label of the tumour site.

        EH:
            The initial helper T cells.

        E:
            The initial cytotoxic T cells.

        A:
            The initial tumour cells.

        ED:
            The initial dendritic cells.

        C:
            The initial IL-2 concentration.

        s1:
            The influx of helper T cells.

        s2:
            The influx of cytotoxic T cells.

        rho1:
            The proliferation constant of helper T cells.

        rho2:
            The proliferation constant of cytotoxic T cells.

        r1:
            The growth rate of helper T cells.

        r2:
            The growth rate of cytotoxic T cells.

        r3:
            The growth rate of the tumour.

        r4:
            The IL-2 secretion rate.

        b1:
            The inverse capacity of helper T cells.

        b2:
            The inverse capacity of cytotoxic T cells.

        b3:
            The inverse capacity of the tumour.

        d1:
            The death rate of helper T cells.

        d2:
            The death rate of cytotoxic T cells.

        d3:
            The degradation rate of IL-2.

        c1:
            The kill rate of tumour cells by cytotoxic T cells.

        c2:
            The kill rate of dendritic cells by cytotoxic T cells.

        c3:
            The consumption rate of IL-2 by cytotoxic T cells.

        dose:
            The injection rate of dendritic cells during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.models.biology.oncology import VaccinatedTumour
        >>> VaccinatedTumour('site').r3
        0.02
    """

    EH: float = state(default=0.02, domain=Domain.NONNEGATIVE)
    E: float = state(default=0.02, domain=Domain.NONNEGATIVE)
    A: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    ED: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    C: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    s1: float = parameter(default=1e-4, domain=Domain.NONNEGATIVE)
    s2: float = parameter(default=1e-4, domain=Domain.NONNEGATIVE)
    rho1: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    rho2: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=1e-2, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=1e-2, domain=Domain.NONNEGATIVE)
    r3: float = parameter(default=0.02, domain=Domain.POSITIVE)
    r4: float = parameter(default=1e-2, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    b2: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    b3: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.005, domain=Domain.NONNEGATIVE)
    d2: float = parameter(default=0.005, domain=Domain.NONNEGATIVE)
    d3: float = parameter(default=1e-2, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    c3: float = parameter(default=1e-7, domain=Domain.NONNEGATIVE)
    dose: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    period: float = 168.0
    width: float = 1.0
    start: float = 0.0
    doses: int = 8

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of T cells, tumour cells, dendritic cells, and IL-2.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        EH, E, A, ED, C = s.EH, s.E, s.A, s.ED, s.C
        vaccine = s.dose * _schedule(s.t, self.period, self.width, self.start, self.doses)
        return [
            Rate(EH, s.s1 + s.rho1 * ED * s.r1 * EH * (1 - s.b1 * EH) - s.d1 * EH),
            Rate(E, s.s2 + s.rho2 * C * (A + ED) * s.r2 * E * (1 - s.b2 * E) - s.d2 * E),
            Rate(A, s.r3 * A * (1 - s.b3 * A) - s.c1 * A * E),
            Rate(ED, vaccine - s.c2 * E * ED),
            Rate(C, s.r4 * EH * ED - s.c3 * E * C - s.d3 * C),
        ]


class CheckpointTumour(Element):
    """Activated T cells, tumour cells, and an anti-PD-1 drug that frees T cells from PD-1-PD-L1 inhibition.

    The T cells follow `dE/dt = (r1 ENC C1 / (C1_50 + C1) + r2 E C2 / (C2_50 + C2)) / (1 + UPP / psi_pp) - d1 E`, with
    the cytokines at their steady states `C1 = C1_ss` and `C2 = C2_ss`, the tumour
    `dA/dt = r3 A (1 - b A) - a_rho A E`, and the drug `dUAP/dt = u_ap - a_rl UFP1 UAP - du2 UAP`. Free PD-1 is
    `UFP1 = max(rho_r - du1 UAP, 0) E`, free PD-L1 `UFL1 = sigma_l (E + c_l A)`, and the complex
    `UPP = sigma UFP1 UFL1`.

    Args:
        label:
            The label of the tumour site.

        E:
            The initial activated T cells.

        A:
            The initial tumour cells.

        UAP:
            The initial drug concentration.

        ENC:
            The naive T cells.

        r1:
            The activation rate of naive T cells by IL-12.

        r2:
            The proliferation rate of T cells by IL-2.

        C1_50:
            The IL-12 at half activation.

        C2_50:
            The IL-2 at half proliferation.

        C1_ss:
            The steady IL-12 concentration.

        C2_ss:
            The steady IL-2 concentration.

        psi_pp:
            The PD-1-PD-L1 complex at half inhibition.

        d1:
            The death rate of T cells.

        r3:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        a_rho:
            The kill rate of tumour cells by T cells.

        a_rl:
            The blocking rate of PD-1 by the drug.

        rho_r:
            The expression of PD-1 on T cells.

        du1:
            The depletion of PD-1 per unit of drug.

        du2:
            The degradation rate of the drug.

        sigma_l:
            The expression of PD-L1 on T cells.

        c_l:
            The expression of PD-L1 on tumour cells relative to T cells.

        sigma:
            The ratio of association to dissociation of PD-1 and PD-L1.

        u_ap:
            The infusion rate of the drug.

    Examples:
        >>> from skcomplex.models.biology.oncology import CheckpointTumour
        >>> CheckpointTumour('site').a_rho
        57.5
    """

    E: float = state(default=6e-3, domain=Domain.NONNEGATIVE)
    A: float = state(default=0.3968, domain=Domain.NONNEGATIVE)
    UAP: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    ENC: float = parameter(default=1e-4, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=8.81, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    C1_50: float = parameter(default=1.5e-10, domain=Domain.POSITIVE)
    C2_50: float = parameter(default=2.37e-11, domain=Domain.POSITIVE)
    C1_ss: float = parameter(default=1.5e-10, domain=Domain.NONNEGATIVE)
    C2_ss: float = parameter(default=2.37e-11, domain=Domain.NONNEGATIVE)
    psi_pp: float = parameter(default=1.365e-18, domain=Domain.POSITIVE)
    d1: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    r3: float = parameter(default=0.18, domain=Domain.POSITIVE)
    b: float = parameter(default=1 / 0.8, domain=Domain.NONNEGATIVE)
    a_rho: float = parameter(default=57.5, domain=Domain.NONNEGATIVE)
    a_rl: float = parameter(default=4.9085e6, domain=Domain.NONNEGATIVE)
    rho_r: float = parameter(default=3.19e-7, domain=Domain.NONNEGATIVE)
    du1: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    du2: float = parameter(default=3.3e-2, domain=Domain.POSITIVE)
    sigma_l: float = parameter(default=3.56e-7, domain=Domain.NONNEGATIVE)
    c_l: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    sigma: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    u_ap: float = parameter(default=1e-10, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of T cells, tumour cells, and the drug.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, A, UAP = s.E, s.A, s.UAP
        expression = s.rho_r - s.du1 * UAP
        free_receptor = where(expression > 0, expression, 0) * E
        free_ligand = s.sigma_l * (E + s.c_l * A)
        complex_ = s.sigma * free_receptor * free_ligand
        activation = s.r1 * s.ENC * _saturation(s.C1_ss, s.C1_50) + s.r2 * E * _saturation(s.C2_ss, s.C2_50)
        return [
            Rate(E, activation / (1 + complex_ / s.psi_pp) - s.d1 * E),
            Rate(A, s.r3 * A * (1 - s.b * A) - s.a_rho * A * E),
            Rate(UAP, s.u_ap - s.a_rl * free_receptor * UAP - s.du2 * UAP),
        ]


class DelayedVaccinatedTumour(Element):
    """Tumour, natural killer cells, naive and mature CD8+ cells, dendritic cells, IL-2, regulatory, and helper cells.

    The equations are those of Padmanabhan et al. (4.13) to (4.21), with the delays `tau1`, `tau2`, and `tau3` of one
    division of natural killer, CD8+, and helper cells. Naive CD8+ cells mature after `7 tau2` and naive helper cells
    after `2 tau3`. The tumour is killed by mature CD8+ cells through `a1 x^5 / (g1 + x^5)` with
    `x = log(E) / log(A)`. The vaccine `u` is `dose` for `width` time units every `period` from `start`, `doses`
    times.

    Args:
        label:
            The label of the tumour site.

        A:
            The initial tumour cells.

        ENK:
            The initial natural killer cells.

        ENC:
            The initial naive CD8+ cells.

        E:
            The initial mature CD8+ cells.

        ED:
            The initial dendritic cells.

        C:
            The initial IL-2 concentration.

        ER:
            The initial regulatory T cells.

        ENH:
            The initial naive helper T cells.

        E1H:
            The initial type 1 helper T cells.

        r1:
            The growth rate of the tumour.

        b:
            The carrying capacity of the tumour.

        a1:
            The maximal kill rate by mature CD8+ cells.

        g1:
            The steepness of the kill by mature CD8+ cells.

        a2:
            The kill rate by natural killer cells.

        s1:
            The influx of natural killer cells.

        d1:
            The death rate of natural killer cells.

        rho1:
            The production rate of natural killer cells by IL-2.

        g2:
            The IL-2 at half production of natural killer cells.

        psi1:
            The inactivation rate of natural killer cells by regulatory cells.

        s2:
            The influx of naive CD8+ cells.

        d2:
            The death rate of naive CD8+ cells.

        rho2:
            The activation rate of naive CD8+ cells by dendritic cells.

        g3:
            The dendritic cells at half activation.

        d3:
            The death rate of mature CD8+ cells.

        rho3:
            The production rate of mature CD8+ cells by dendritic cells.

        m1:
            The proliferation rate of mature CD8+ cells with dendritic cells.

        rho4:
            The proliferation rate of mature CD8+ cells by IL-2.

        g4:
            The squared IL-2 at half proliferation.

        psi2:
            The inactivation rate of mature CD8+ cells by regulatory cells.

        d4:
            The death rate of dendritic cells.

        kappa1:
            The production of dendritic cells from tumour antigen.

        g5:
            The tumour at half production of dendritic cells.

        s3:
            The influx of IL-2.

        d5:
            The decay rate of IL-2.

        rho5:
            The secretion rate of IL-2 by mature CD8+ cells.

        g6:
            The squared mature CD8+ cells at half secretion.

        rho6:
            The secretion rate of IL-2 by type 1 helper cells.

        g7:
            The type 1 helper cells at half secretion.

        s4:
            The influx of regulatory cells.

        d6:
            The death rate of regulatory cells.

        rho7:
            The production rate of regulatory cells by the tumour.

        g8:
            The squared tumour at half production of regulatory cells.

        rho8:
            The production rate of regulatory cells by IL-2.

        g9:
            The IL-2 at half production of regulatory cells.

        s5:
            The influx of naive helper cells.

        d7:
            The death rate of naive helper cells.

        rho9:
            The activation rate of naive helper cells by dendritic cells.

        d8:
            The death rate of type 1 helper cells.

        m2:
            The maturation rate of naive helper cells.

        psi3:
            The inactivation rate of type 1 helper cells by regulatory cells.

        rho10:
            The proliferation rate of type 1 helper cells with dendritic cells.

        tau1:
            The division time of natural killer cells.

        tau2:
            The division time of CD8+ cells.

        tau3:
            The division time of helper cells.

        dose:
            The injection rate of dendritic cells during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.models.biology.oncology import DelayedVaccinatedTumour
        >>> DelayedVaccinatedTumour('site').tau1
        1.26
    """

    A: float = state(default=5e5, domain=Domain.POSITIVE)
    ENK: float = state(default=4.4e4, domain=Domain.NONNEGATIVE)
    ENC: float = state(default=1.2e5, domain=Domain.NONNEGATIVE)
    E: float = state(default=10.0, domain=Domain.POSITIVE)
    ED: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    C: float = state(default=48.0, domain=Domain.NONNEGATIVE)
    ER: float = state(default=1.1e6, domain=Domain.NONNEGATIVE)
    ENH: float = state(default=1.26e5, domain=Domain.NONNEGATIVE)
    E1H: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.16, domain=Domain.POSITIVE)
    b: float = parameter(default=5e9, domain=Domain.POSITIVE)
    a1: float = parameter(default=6.9, domain=Domain.NONNEGATIVE)
    g1: float = parameter(default=3.5, domain=Domain.POSITIVE)
    a2: float = parameter(default=7.13e-10, domain=Domain.NONNEGATIVE)
    s1: float = parameter(default=1.32e4, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=4.12e-2, domain=Domain.NONNEGATIVE)
    rho1: float = parameter(default=6.68e-2, domain=Domain.NONNEGATIVE)
    g2: float = parameter(default=2.5e4, domain=Domain.POSITIVE)
    psi1: float = parameter(default=1e-7, domain=Domain.NONNEGATIVE)
    s2: float = parameter(default=5e5, domain=Domain.NONNEGATIVE)
    d2: float = parameter(default=0.03, domain=Domain.NONNEGATIVE)
    rho2: float = parameter(default=1.2, domain=Domain.NONNEGATIVE)
    g3: float = parameter(default=10**5.4, domain=Domain.POSITIVE)
    d3: float = parameter(default=0.4, domain=Domain.NONNEGATIVE)
    rho3: float = parameter(default=8.98e-7, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=0.38, domain=Domain.NONNEGATIVE)
    rho4: float = parameter(default=0.19, domain=Domain.NONNEGATIVE)
    g4: float = parameter(default=6.25e6, domain=Domain.POSITIVE)
    psi2: float = parameter(default=4.3e-7, domain=Domain.NONNEGATIVE)
    d4: float = parameter(default=0.237, domain=Domain.NONNEGATIVE)
    kappa1: float = parameter(default=2.438e4, domain=Domain.NONNEGATIVE)
    g5: float = parameter(default=1e5, domain=Domain.POSITIVE)
    s3: float = parameter(default=280.5, domain=Domain.NONNEGATIVE)
    d5: float = parameter(default=5.5, domain=Domain.NONNEGATIVE)
    rho5: float = parameter(default=1.1, domain=Domain.NONNEGATIVE)
    g6: float = parameter(default=1e11, domain=Domain.POSITIVE)
    rho6: float = parameter(default=4.99, domain=Domain.NONNEGATIVE)
    g7: float = parameter(default=1e5, domain=Domain.POSITIVE)
    s4: float = parameter(default=3.212e5, domain=Domain.NONNEGATIVE)
    d6: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    rho7: float = parameter(default=0.26, domain=Domain.NONNEGATIVE)
    g8: float = parameter(default=4.5e17, domain=Domain.POSITIVE)
    rho8: float = parameter(default=0.399, domain=Domain.NONNEGATIVE)
    g9: float = parameter(default=10**4.4, domain=Domain.POSITIVE)
    s5: float = parameter(default=5.5e5, domain=Domain.NONNEGATIVE)
    d7: float = parameter(default=0.03, domain=Domain.NONNEGATIVE)
    rho9: float = parameter(default=1.05, domain=Domain.NONNEGATIVE)
    d8: float = parameter(default=0.44, domain=Domain.NONNEGATIVE)
    m2: float = parameter(default=0.45, domain=Domain.NONNEGATIVE)
    psi3: float = parameter(default=8e-7, domain=Domain.NONNEGATIVE)
    rho10: float = parameter(default=1.4, domain=Domain.NONNEGATIVE)
    tau1: float = parameter(default=1.26, domain=Domain.POSITIVE)
    tau2: float = parameter(default=0.45, domain=Domain.POSITIVE)
    tau3: float = parameter(default=0.47, domain=Domain.POSITIVE)
    dose: float = parameter(default=1e6, domain=Domain.NONNEGATIVE)
    period: float = 7.0
    width: float = 1.0
    start: float = 0.0
    doses: int = 4

    def laws(self, s: Symbols) -> list[Law]:
        """Return the delayed balances of the tumour and the eight immune populations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, ENK, ENC, E, ED, C, ER, ENH, E1H = s.A, s.ENK, s.ENC, s.E, s.ED, s.C, s.ER, s.ENH, s.E1H
        ratio = (log(E) / log(A)) ** 5
        vaccine = s.dose * _schedule(s.t, self.period, self.width, self.start, self.doses)
        C1, ED2, E2, C2 = delayed(C, s.tau1), delayed(ED, s.tau2), delayed(E, s.tau2), delayed(C, s.tau2)
        ER3, C3, A3, ED3, E3 = (delayed(x, s.tau3) for x in (ER, C, A, ED, E))
        ED7, ENC7 = (delayed(x, 7 * s.tau2, 7 * self.tau2) for x in (ED, ENC))
        ED6, ENH6 = (delayed(x, 2 * s.tau3, 2 * self.tau3) for x in (ED, ENH))
        return [
            Rate(A, s.r1 * A * log(s.b / A) - s.a1 * ratio / (ratio + s.g1) * A - s.a2 * ENK * A),
            Rate(ENK, s.s1 - s.d1 * ENK + s.rho1 * delayed(ENK, s.tau1) * C1 / (s.g2 + C1) - s.psi1 * ENK * ER),
            Rate(ENC, s.s2 - s.d2 * ENC - s.rho2 * ED * ENC / (s.g3 + ED)),
            Rate(
                E,
                -s.d3 * E
                + s.rho3 * ENC7 * ED7 / (s.g3 + ED7)
                + s.m1 * ED2 * E2 / (s.g3 + ED2)
                + s.rho4 * E2 * C2**2 / (s.g4 + C2**2)
                - s.psi2 * E * ER,
            ),
            Rate(ED, -s.d4 * ED + s.kappa1 * A / (s.g5 + A) + vaccine),
            Rate(C, s.s3 - s.d5 * C + s.rho5 * E**2 * C / (s.g6 + E**2) + s.rho6 * E1H * C / (s.g7 + E1H)),
            Rate(ER, s.s4 - s.d6 * ER + s.rho7 * ER3 * A3**2 / (s.g8 + A3**2) + s.rho8 * ER3 * C3 / (s.g9 + C3)),
            Rate(ENH, s.s5 - s.d7 * ENH - s.rho9 * ED * ENH / (s.g3 + ED)),
            Rate(
                E1H,
                -s.d8 * E1H + s.m2 * ED6 * ENH6 / (s.g3 + ED6) - s.psi3 * ER * E1H + s.rho10 * ED3 * E3 / (s.g3 + ED3),
            ),
        ]


class SuppressedTumour(Element):
    """Tumour, dendritic cells, proliferating and active T cells, IL-2, IL-10, and TGF-beta.

    The equations are those of Padmanabhan et al. (4.22) to (4.33). The tumour grows by the smooth minimum
    `A / ((1 / r1)^p + (A^(1 - n) / r2)^p)^(1/p)` of exponential and power growth and is killed through its fraction
    `A* = A / (1 + (A^(1 - q) / g3)^p)^(1/p)` accessible to the cytotoxic T cells `E`, at the rate
    `a1 A* E / (E + A* / g1 + ER / g2) / (1 + C3 / C3_50)`. Antigen matures unlicensed dendritic cells, which helper
    memory cells license, and dendritic cells recruit the memory pools `EMC`, `EMH`, and `EMR` into proliferating
    cytotoxic, helper, and regulatory cells, which IL-2 matures. IL-10 and TGF-beta, from regulatory and tumour cells,
    suppress maturation, IL-2 production, and killing, and TGF-beta converts helper into regulatory cells.

    Args:
        label:
            The label of the tumour site.

        A:
            The initial tumour cells.

        EUD:
            The initial unlicensed dendritic cells.

        ED:
            The initial licensed dendritic cells.

        EPC:
            The initial proliferating cytotoxic T cells.

        E:
            The initial active cytotoxic T cells.

        EPH:
            The initial proliferating helper T cells.

        EH:
            The initial active helper T cells.

        EPR:
            The initial proliferating regulatory T cells.

        ER:
            The initial active regulatory T cells.

        C1:
            The initial IL-2 concentration.

        C2:
            The initial IL-10 concentration.

        C3:
            The initial TGF-beta concentration.

        r1:
            The exponential growth rate of the tumour.

        r2:
            The power growth coefficient of the tumour.

        n:
            The exponent of the power growth.

        p:
            The smoothness of the transitions.

        q:
            The exponent of the accessible fraction of a large tumour.

        a1:
            The maximal kill rate of tumour cells.

        g1:
            The ratio of tumour to cytotoxic cells at half kill.

        g2:
            The ratio of regulatory to cytotoxic cells at half kill.

        g3:
            The size scale of the accessible fraction.

        C3_50:
            The TGF-beta at half kill.

        alpha_a:
            The antigenicity of the tumour.

        C2_50:
            The IL-10 at half dendritic maturation.

        ER_50:
            The regulatory cells at half dendritic maturation.

        lambda_a:
            The licensing rate of dendritic cells.

        d1:
            The death rate of unlicensed dendritic cells.

        d2:
            The death rate of licensed dendritic cells.

        a2:
            The recruitment rate of proliferating cytotoxic cells.

        g4:
            The steepness of the recruitment by dendritic cells.

        d3:
            The death rate of proliferating cytotoxic cells.

        a3:
            The maturation rate of cytotoxic cells.

        EC1_50:
            The IL-2 at half maturation.

        EC3_50:
            The TGF-beta at half maturation.

        d4:
            The death rate of active cytotoxic cells.

        a4:
            The recruitment rate of proliferating helper cells.

        d5:
            The death rate of proliferating helper cells.

        a5:
            The maturation rate of helper cells.

        a6:
            The conversion rate of helper into regulatory cells.

        g5:
            The TGF-beta at half conversion.

        d6:
            The death rate of active helper cells.

        a7:
            The recruitment rate of proliferating regulatory cells.

        d7:
            The death rate of proliferating regulatory cells.

        a8:
            The maturation rate of regulatory cells.

        d8:
            The death rate of active regulatory cells.

        a9:
            The IL-2 production by helper cells.

        CC2_50:
            The IL-10 at half suppression of IL-2.

        CC3_50:
            The TGF-beta at half suppression of IL-2.

        tau1:
            The lifetime of IL-2.

        r3:
            The IL-10 production by regulatory cells.

        r4:
            The IL-10 production by tumour cells.

        tau2:
            The lifetime of IL-10.

        r5:
            The TGF-beta production by regulatory cells.

        r6:
            The TGF-beta production by tumour cells.

        tau3:
            The lifetime of TGF-beta.

        EMC:
            The memory cytotoxic T cells.

        EMH:
            The memory helper T cells.

        EMR:
            The memory regulatory T cells.

    Examples:
        >>> from skcomplex.models.biology.oncology import SuppressedTumour
        >>> SuppressedTumour('site').lambda_a
        0.5
    """

    A: float = state(default=1.0, domain=Domain.POSITIVE)
    EUD: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    ED: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    EPC: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    E: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    EPH: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    EH: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    EPR: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    ER: float = state(default=1e-5, domain=Domain.NONNEGATIVE)
    C1: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    C2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    C3: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.333, domain=Domain.POSITIVE)
    r2: float = parameter(default=333.0, domain=Domain.POSITIVE)
    n: float = parameter(default=0.5, domain=Domain.POSITIVE)
    p: float = parameter(default=3.0, domain=Domain.POSITIVE)
    q: float = parameter(default=2 / 3, domain=Domain.POSITIVE)
    a1: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)
    g1: float = parameter(default=1.2, domain=Domain.POSITIVE)
    g2: float = parameter(default=11.0, domain=Domain.POSITIVE)
    g3: float = parameter(default=100.0, domain=Domain.POSITIVE)
    C3_50: float = parameter(default=3.5, domain=Domain.POSITIVE)
    alpha_a: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    C2_50: float = parameter(default=0.4, domain=Domain.POSITIVE)
    ER_50: float = parameter(default=2e7, domain=Domain.POSITIVE)
    lambda_a: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.14, domain=Domain.NONNEGATIVE)
    d2: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=23.0, domain=Domain.NONNEGATIVE)
    g4: float = parameter(default=0.33, domain=Domain.POSITIVE)
    d3: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=16.0, domain=Domain.NONNEGATIVE)
    EC1_50: float = parameter(default=0.3, domain=Domain.POSITIVE)
    EC3_50: float = parameter(default=2.9, domain=Domain.POSITIVE)
    d4: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    a4: float = parameter(default=9.9, domain=Domain.NONNEGATIVE)
    d5: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    a5: float = parameter(default=1.9, domain=Domain.NONNEGATIVE)
    a6: float = parameter(default=0.022, domain=Domain.NONNEGATIVE)
    g5: float = parameter(default=1.7, domain=Domain.POSITIVE)
    d6: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    a7: float = parameter(default=5.1, domain=Domain.NONNEGATIVE)
    d7: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    a8: float = parameter(default=2.1, domain=Domain.NONNEGATIVE)
    d8: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    a9: float = parameter(default=1.7e-5, domain=Domain.NONNEGATIVE)
    CC2_50: float = parameter(default=0.75, domain=Domain.POSITIVE)
    CC3_50: float = parameter(default=0.9, domain=Domain.POSITIVE)
    tau1: float = parameter(default=0.08, domain=Domain.POSITIVE)
    r3: float = parameter(default=1.4e-8, domain=Domain.NONNEGATIVE)
    r4: float = parameter(default=1.3e-10, domain=Domain.NONNEGATIVE)
    tau2: float = parameter(default=0.05, domain=Domain.POSITIVE)
    r5: float = parameter(default=1.8e-8, domain=Domain.NONNEGATIVE)
    r6: float = parameter(default=1.1e-7, domain=Domain.NONNEGATIVE)
    tau3: float = parameter(default=0.07, domain=Domain.POSITIVE)
    EMC: float = parameter(default=1e4, domain=Domain.NONNEGATIVE)
    EMH: float = parameter(default=1e4, domain=Domain.POSITIVE)
    EMR: float = parameter(default=1e4, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of the tumour, the immune cells, and the three cytokines.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, EUD, ED, EPC, E, EPH, EH, EPR, ER, C1, C2, C3 = (
            s.A,
            s.EUD,
            s.ED,
            s.EPC,
            s.E,
            s.EPH,
            s.EH,
            s.EPR,
            s.ER,
            s.C1,
            s.C2,
            s.C3,
        )
        growth = A / ((1 / s.r1) ** s.p + (A ** (1 - s.n) / s.r2) ** s.p) ** (1 / s.p)
        accessible = A / (1 + (A ** (1 - s.q) / s.g3) ** s.p) ** (1 / s.p)
        kill = s.a1 * accessible * E / (E + accessible / s.g1 + ER / s.g2) / (1 + C3 / s.C3_50)
        licensing = s.lambda_a * EUD / (1 + EUD / s.EMH)
        memory = s.EMC + s.EMH + s.EMR
        maturation = _saturation(C1, s.EC1_50) / (1 + C3 / s.EC3_50)
        conversion = s.a6 * EH * C3 / (s.g5 + C3)
        return [
            Rate(A, growth - kill),
            Rate(EUD, s.alpha_a * accessible / ((1 + C2 / s.C2_50) * (1 + ER / s.ER_50)) - licensing - s.d1 * EUD),
            Rate(ED, licensing - s.d2 * ED),
            Rate(EPC, s.a2 * s.EMC * ED / (ED + s.g4 * memory) - s.d3 * EPC),
            Rate(E, s.a3 * EPC * maturation - s.d4 * E),
            Rate(EPH, s.a4 * s.EMH * (ED + EUD) / (ED + EUD + s.g4 * memory) - s.d5 * EPH),
            Rate(EH, s.a5 * EPH * maturation - conversion - s.d6 * EH),
            Rate(EPR, s.a7 * s.EMR * ED / (ED + s.g4 * memory) - s.d7 * EPR),
            Rate(ER, s.a8 * EPR * _saturation(C1, s.EC1_50) + conversion - s.d8 * ER),
            Rate(C1, s.a9 * EPH / ((1 + C3 / s.CC3_50) * (1 + C2 / s.CC2_50)) - C1 / s.tau1),
            Rate(C2, s.r3 * ER + s.r4 * A - C2 / s.tau2),
            Rate(C3, s.r5 * ER + s.r6 * A - C3 / s.tau3),
        ]


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'adoptive therapy', 'IL-2', 'oscillations'),
    references=(BOOK_REFERENCE, 'Kirschner and Panetta 1998'),
    defaults={'until': 1000.0, 'dt': 1.0, 'solver': STIFF_SOLVER},
)
def kirschner_panetta(
    E: float = 1.0,
    A: float = 1.0,
    C: float = 1.0,
    *,
    alpha_a: float = 0.02,
    r1: float = 0.18,
    c1: float = 1.0,
    a1: float = 500.0,
    a2: float = 7e7,
    u1: float = 0.0,
    u2: float = 0.0,
) -> System:
    """Return the Kirschner-Panetta model of effector cells, tumour cells, and IL-2 under adoptive therapy and IL-2.

    Untreated, a tumour of antigenicity in the oscillatory range, such as the default 0.02, and the effector cells
    oscillate in long cycles in which the tumour nearly vanishes and regrows to about `4 10^5`. The adoptive therapy
    adds `a1 u1` effector cells per day and the IL-2 injection `a2 u2` IL-2 per day. With `u1` and `u2` above 1, the
    book's treatment factors `a1 = 500` and `a2 = 7 10^7` are exceeded and the tumour is cleared. The populations are
    in volume units, the inputs in volume per day, and time in days, and the initial state is the book's.

    Args:
        E:
            The initial effector cells.

        A:
            The initial tumour cells.

        C:
            The initial IL-2 concentration.

        alpha_a:
            The antigenicity of the tumour, in the book's range 0 to 0.05.

        r1:
            The growth rate of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

        a1:
            The effector cells per unit of adoptive therapy.

        a2:
            The IL-2 per unit of IL-2 injection.

        u1:
            The adoptive therapy input.

        u2:
            The IL-2 injection input.

    Returns:
        system:
            The model, with the effector cells `site.E`, the tumour cells `site.A`, and the IL-2 `site.C`.

    Examples:
        >>> from skcomplex.models.biology.oncology import kirschner_panetta
        >>> kirschner_panetta(u1=1.0).elements[0].u1
        1.0
    """
    site = KirschnerPanettaTumour('site', E=E, A=A, C=C, alpha_a=alpha_a, r1=r1, c1=c1, a1=a1, a2=a2, u1=u1, u2=u2)
    return System(components=[site])


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'vaccine', 'dendritic cells', 'dosing'),
    references=(BOOK_REFERENCE, 'Castiglione and Piccoli 2006'),
    defaults={'until': 2880.0, 'dt': 2.0, 'solver': STIFF_SOLVER},
)
def dendritic_cell_vaccine(
    A: float = 0.1,
    *,
    rho2: float = 10.0,
    r4: float = 1e-2,
    dose: float = 0.5,
    period: float = 168.0,
    width: float = 1.0,
    start: float = 0.0,
    doses: int = 8,
) -> System:
    """Return helper and cytotoxic T cells, a tumour, dendritic cells, and IL-2 under a dendritic-cell vaccine.

    Vaccinated dendritic cells drive the proliferation of helper cells, which secrete IL-2, and IL-2 with the antigen
    drives the proliferation of cytotoxic cells, which kill the tumour. The book gives no initial state or schedule.
    The T cells start at their resting level `s / d = 0.02`, the dendritic cells and IL-2 at 0, and the vaccine is a
    one-hour injection of `dose` per hour every week, 8 times. Without vaccine the tumour settles at
    `1 - c1 s2 / (d2 r3) = 0.9`. The populations are in cells per mm³ scaled by the carrying capacity, IL-2 in
    pg/mm³, and time in hours.

    Args:
        A:
            The initial tumour cells.

        rho2:
            The proliferation constant of cytotoxic T cells.

        r4:
            The IL-2 secretion rate.

        dose:
            The injection rate of dendritic cells during a dose, 0 for no vaccine.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

    Returns:
        system:
            The model, with the helper cells `site.EH`, the cytotoxic cells `site.E`, the tumour `site.A`, the
            dendritic cells `site.ED`, and the IL-2 `site.C`.

    Examples:
        >>> from skcomplex.models.biology.oncology import dendritic_cell_vaccine
        >>> dendritic_cell_vaccine(dose=0.0).elements[0].dose
        0.0
    """
    site = VaccinatedTumour(
        'site',
        A=A,
        rho2=rho2,
        r4=r4,
        dose=dose,
        period=period,
        width=width,
        start=start,
        doses=doses,
    )
    return System(components=[site])


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'checkpoint inhibitor', 'PD-1', 'dosing'),
    references=(BOOK_REFERENCE, 'Nikolopoulou et al. 2018'),
    defaults={'until': 150.0, 'dt': 0.5, 'solver': STIFF_SOLVER},
)
def anti_pd1_checkpoint(
    E: float = 6e-3,
    A: float = 0.3968,
    *,
    u_ap: float = 1e-10,
    sigma: float = 5.0,
    ENC: float = 1e-4,
    r3: float = 0.18,
    b: float = 1 / 0.8,
    c_l: float = 1.0,
) -> System:
    """Return activated T cells, a tumour, and an anti-PD-1 drug that frees the T cells from checkpoint inhibition.

    The PD-1-PD-L1 complex inhibits the activation and proliferation of T cells, and the drug lowers the free PD-1 on
    T cells. The cytokines are at their steady states, as in the book. The book gives neither the naive T cells `ENC`
    nor the association ratio `sigma`, which are set so that the T cells settle below the level that controls the
    tumour. The ranged parameters take the low ends of the book's ranges, and the drug is the book's continuous
    infusion of `10^-10` per day. A larger infusion frees more T cells and lowers the tumour. The book's free PD-1
    `(rho_r - du1 UAP) E` turns negative once the drug exceeds `rho_r / du1`, and it is held at 0 there. With all PD-1
    blocked nothing limits the T cells, which grow without bound and clear the tumour. Cells and the drug are in g/cm³
    and time in days.

    Args:
        E:
            The initial activated T cells.

        A:
            The initial tumour cells.

        u_ap:
            The infusion rate of the drug, 0 for no treatment.

        sigma:
            The ratio of association to dissociation of PD-1 and PD-L1.

        ENC:
            The naive T cells.

        r3:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c_l:
            The expression of PD-L1 on tumour cells relative to T cells.

    Returns:
        system:
            The model, with the T cells `site.E`, the tumour `site.A`, and the drug `site.UAP`.

    Examples:
        >>> from skcomplex.models.biology.oncology import anti_pd1_checkpoint
        >>> anti_pd1_checkpoint(u_ap=0.0).elements[0].u_ap
        0.0
    """
    site = CheckpointTumour('site', E=E, A=A, u_ap=u_ap, sigma=sigma, ENC=ENC, r3=r3, b=b, c_l=c_l)
    return System(components=[site])


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'delay', 'vaccine', 'dendritic cells', 'regulatory T cells'),
    references=(BOOK_REFERENCE, 'Qomlaqi et al. 2017'),
    defaults={'until': 30.0, 'dt': 0.1, 'solver': STIFF_SOLVER},
)
def delayed_dc_vaccine(
    *,
    dose: float = 1e6,
    period: float = 7.0,
    width: float = 1.0,
    start: float = 0.0,
    doses: int = 4,
    tau1: float = 1.26,
    tau2: float = 0.45,
    tau3: float = 0.47,
) -> System:
    """Return a tumour and eight innate and adaptive immune populations, with division delays, under a vaccine.

    Natural killer cells divide in `tau1`, CD8+ cells mature after seven divisions `7 tau2`, and naive helper cells
    become type 1 helper cells after two divisions `2 tau3`. Regulatory T cells, driven by the tumour and IL-2, inhibit
    natural killer, CD8+, and helper cells. The book's growth term `m1 ED E` of mature CD8+ cells has no saturation and
    makes them diverge within a day, so it saturates in the dendritic cells as `m1 ED E / (g3 + ED)`, like every other
    term the dendritic cells drive. The book gives no schedule, and the vaccine is a one-day injection of `dose`
    dendritic cells per day every week, 4 times. The parameters and initial state are the book's, and with them the
    regulatory T cells hold the adaptive response down, so the tumour grows toward its capacity `b`. Cells are counts,
    IL-2 is in IU/L, and time in days.

    Args:
        dose:
            The injection rate of dendritic cells during a dose, 0 for no vaccine.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        doses:
            The number of doses.

        tau1:
            The division time of natural killer cells.

        tau2:
            The division time of CD8+ cells.

        tau3:
            The division time of helper cells.

    Returns:
        system:
            The model, with the tumour `site.A`, the natural killer cells `site.ENK`, the naive and mature CD8+ cells
            `site.ENC` and `site.E`, the dendritic cells `site.ED`, the IL-2 `site.C`, the regulatory cells `site.ER`,
            and the naive and type 1 helper cells `site.ENH` and `site.E1H`.

    Examples:
        >>> from skcomplex.models.biology.oncology import delayed_dc_vaccine
        >>> delayed_dc_vaccine(tau2=0.5).elements[0].tau2
        0.5
    """
    site = DelayedVaccinatedTumour(
        'site',
        dose=dose,
        period=period,
        width=width,
        start=start,
        doses=doses,
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
    )
    return System(components=[site])


@register_model(
    field='biology/oncology',
    tags=('immunotherapy', 'ode', 'immunosuppression', 'cytokines', 'antigenicity'),
    references=(BOOK_REFERENCE, 'Robertson-Tessi, El-Kareh, and Goriely 2012'),
    defaults={'until': 200.0, 'dt': 0.5, 'solver': STIFF_SOLVER},
)
def tumour_immune_suppression(
    *,
    alpha_a: float = 1e-3,
    r1: float = 0.333,
    r2: float = 333.0,
    EMC: float = 1e4,
    EMH: float = 1e4,
    EMR: float = 1e4,
) -> System:
    """Return a tumour against dendritic, cytotoxic, helper, and regulatory T cells with IL-2, IL-10, and TGF-beta.

    A tumour of one cell grows exponentially, then as a power of its size, and only its accessible fraction is killed.
    Its antigen matures dendritic cells, which recruit T cells from the memory pools, and the regulatory cells with
    IL-10 and TGF-beta suppress the response. A weakly antigenic tumour escapes, a moderately antigenic one oscillates,
    and a strongly antigenic one is held to a few cells. The book's growth rows are garbled, and the exponential rate
    is read as 0.333 per day and the power coefficient as 333 cells^(1/2) per day, within the book's range 100 to
    1000, so the growth turns from exponential to power at `10^6` cells. The book gives no memory pools, which are
    taken as constant at `10^4` cells each. The other parameters are the book's, the initial immune cells are
    `10^-5`, cells are counts, cytokines in ng/ml, and time in days.

    Args:
        alpha_a:
            The antigenicity of the tumour, in the book's range `10^-5` to `10^5`.

        r1:
            The exponential growth rate of the tumour.

        r2:
            The power growth coefficient of the tumour.

        EMC:
            The memory cytotoxic T cells.

        EMH:
            The memory helper T cells.

        EMR:
            The memory regulatory T cells.

    Returns:
        system:
            The model, with the tumour `site.A`, the dendritic cells `site.EUD` and `site.ED`, the T cells `site.EPC`,
            `site.E`, `site.EPH`, `site.EH`, `site.EPR`, and `site.ER`, and the cytokines `site.C1`, `site.C2`, and
            `site.C3`.

    Examples:
        >>> from skcomplex.models.biology.oncology import tumour_immune_suppression
        >>> tumour_immune_suppression(alpha_a=1.0).elements[0].alpha_a
        1.0
    """
    site = SuppressedTumour('site', alpha_a=alpha_a, r1=r1, r2=r2, EMC=EMC, EMH=EMH, EMR=EMR)
    return System(components=[site])
