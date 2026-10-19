"""Oncology models: prostate and breast cancers under hormone therapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import log

from ....core.base import Domain, Expression, ValidationError, parameter, state
from ....core.model import Element, Event, Law, Rate, Symbols, System, pulse_train
from ....core.runtime import register_model
from ....core.solvers import ScipyODE

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


def _prostate_volume(s: Symbols) -> Expression:
    """Return the prostate volume in cm³ at the age of the patient, `v1 a^phi / (v2^phi + a^phi) + v3`."""
    age = s.age + s.t / 8760
    return s.v1 * age**s.phi / (s.v2**s.phi + age**s.phi) + s.v3


def _receptor_growth(s: Symbols, psi1: Expression, phi1: Expression) -> Expression:
    """Return the growth rate that the receptors activated by DHT and testosterone drive."""
    return psi1 * s.UD**2 / (s.psi2**2 + s.UD**2) + phi1 * s.UT**2 / (s.phi2**2 + s.UT**2)


def _receptor_death(s: Symbols, psi3: Expression) -> Expression:
    """Return the death rate that the activated receptors lower."""
    return psi3 / (s.psi4 + s.UD + s.UT)


def _quota_rate(s: Symbols, Q: Expression, q: Expression) -> Expression:
    """Return the uptake of androgen into a quota less its use for growth and its degradation."""
    uptake = s.vm * (s.qm - Q) / (s.qm - q) * s.U / (s.U + s.v50)
    return uptake - s.rm * (Q - q) - s.d3 * Q


class AndrogenSuppression(Element):
    """Androgen-dependent and androgen-independent prostate cancer cells under continuous androgen suppression.

    The dependent cells follow `dAD/dt = r1 U (1 - b (AD + c1 AI)) AD - (d1 + m1)(1 - U / U0) AD`, the independent
    cells `dAI/dt = r2 (1 - b (AI + c2 AD)) AI + m1 (1 - U / U0) AD`, and the serum androgen
    `du/dt = -du (U - U0) - U0 du a`, which the drug efficacy `a` lowers to `U0 (1 - a)`.

    Args:
        label:
            The label of the tumour.

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
        >>> from skcomplex.models.biology.oncology import AndrogenSuppression
        >>> AndrogenSuppression('tumour', a=0.6).a
        0.6
    """

    AD: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    AI: float = state(default=0.14, domain=Domain.NONNEGATIVE)
    U: float = state(default=20.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.025, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.006, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=0.091e-9, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.064, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=5e-5, domain=Domain.NONNEGATIVE)
    U0: float = parameter(default=20.0, domain=Domain.POSITIVE)
    du: float = parameter(default=0.08, domain=Domain.NONNEGATIVE)
    a: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.AD, s.r1 * s.U * (1 - s.b * (s.AD + s.c1 * s.AI)) * s.AD - (s.d1 + s.m1) * deprivation * s.AD),
            Rate(s.AI, s.r2 * (1 - s.b * (s.AI + s.c2 * s.AD)) * s.AI + s.m1 * deprivation * s.AD),
            Rate(s.U, -s.du * (s.U - s.U0) - s.U0 * s.du * s.a),
        ]


class IntermittentSuppression(Element):
    """Prostate cancer cells under androgen suppression that PSA thresholds switch on and off.

    The dependent cells follow `dAD/dt = r1 (k1 + (1 - k1) U / (U + k2)) AD - d1 (k3 + (1 - k3) U / (U + k4)) AD
    - m1 (1 - U / U0) AD`, the independent cells `dAI/dt = m1 (1 - U / U0) AD + r2 (1 - U / U0) AI - d2 AI`, and
    the serum androgen `du/dt = -du (U - U0) - U0 du u`. The therapy `u` turns 1 when the PSA `x1 AD + x2 AI` rises
    through `h1` and 0 when it falls through `h0`.

    Args:
        label:
            The label of the tumour.

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
        >>> from skcomplex.models.biology.oncology import IntermittentSuppression
        >>> IntermittentSuppression('tumour').h1
        15.0
    """

    AD: float = state(default=5.0, domain=Domain.NONNEGATIVE)
    AI: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    U: float = state(default=30.0, domain=Domain.NONNEGATIVE)
    u: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.0204, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.0242, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.0076, domain=Domain.NONNEGATIVE)
    d2: float = parameter(default=0.0168, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=5e-5, domain=Domain.NONNEGATIVE)
    k1: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    k2: float = parameter(default=2.0, domain=Domain.POSITIVE)
    k3: float = parameter(default=8.0, domain=Domain.NONNEGATIVE)
    k4: float = parameter(default=0.5, domain=Domain.POSITIVE)
    U0: float = parameter(default=30.0, domain=Domain.POSITIVE)
    du: float = parameter(default=0.08, domain=Domain.NONNEGATIVE)
    x1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    x2: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    h0: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    h1: float = parameter(default=15.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.AD, (growth - death - s.m1 * deprivation) * s.AD),
            Rate(s.AI, s.m1 * deprivation * s.AD + (s.r2 * deprivation - s.d2) * s.AI),
            Rate(s.U, -s.du * (s.U - s.U0) - s.U0 * s.du * s.u),
            Event(psa - s.h1, {s.u: 1}, direction=1, label='on'),
            Event(psa - s.h0, {s.u: 0}, direction=-1, label='off'),
        ]


class CellQuotaProstate(Element):
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
        >>> from skcomplex.models.biology.oncology import CellQuotaProstate
        >>> CellQuotaProstate('tumour').qm
        5.0
    """

    AD: float = state(default=15.0, domain=Domain.NONNEGATIVE)
    AI: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Q1: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    Q2: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    P: float = state(default=3.4, domain=Domain.NONNEGATIVE)
    U: float = state(default=15.0, domain=Domain.NONNEGATIVE)
    rm: float = parameter(default=0.025, domain=Domain.NONNEGATIVE)
    q1: float = parameter(default=0.3, domain=Domain.POSITIVE)
    q2: float = parameter(default=0.15, domain=Domain.POSITIVE)
    d1: float = parameter(default=0.015, domain=Domain.NONNEGATIVE)
    d2: float = parameter(default=0.015, domain=Domain.NONNEGATIVE)
    d3: float = parameter(default=0.09, domain=Domain.NONNEGATIVE)
    qm: float = parameter(default=5.0, domain=Domain.POSITIVE)
    vm: float = parameter(default=0.275, domain=Domain.NONNEGATIVE)
    v50: float = parameter(default=4.0, domain=Domain.POSITIVE)
    mx1: float = parameter(default=0.00015, domain=Domain.NONNEGATIVE)
    mx2: float = parameter(default=0.0001, domain=Domain.NONNEGATIVE)
    m501: float = parameter(default=0.08, domain=Domain.POSITIVE)
    m502: float = parameter(default=1.7, domain=Domain.POSITIVE)
    n: float = parameter(default=3.0, domain=Domain.POSITIVE)
    r0: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    rho1: float = parameter(default=1.3, domain=Domain.POSITIVE)
    rho2: float = parameter(default=1.1, domain=Domain.POSITIVE)
    theta: float = parameter(default=3.0, domain=Domain.POSITIVE)
    du: float = parameter(default=0.08, domain=Domain.NONNEGATIVE)
    U0: float = parameter(default=15.0, domain=Domain.NONNEGATIVE)
    Umin: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    du1: float = parameter(default=0.08, domain=Domain.NONNEGATIVE)
    period: float = 500.0
    width: float = 250.0

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.AD, s.rm * (1 - s.q1 / s.Q1) * s.AD - s.d1 * s.AD - to_independent + to_dependent),
            Rate(s.AI, s.rm * (1 - s.q2 / s.Q2) * s.AI - s.d2 * s.AI + to_independent - to_dependent),
            Rate(s.Q1, _quota_rate(s, s.Q1, s.q1)),
            Rate(s.Q2, _quota_rate(s, s.Q2, s.q2)),
            Rate(s.P, s.r0 * (s.AD + s.AI) + produced - s.du * s.P),
            Rate(s.U, s.du1 * (androgen - s.U)),
        ]


class AndrogenReceptorProstate(Element):
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
        >>> from skcomplex.models.biology.oncology import AndrogenReceptorProstate
        >>> AndrogenReceptorProstate('prostate').KP
        800.0
    """

    N: float = state(default=1.5, domain=Domain.NONNEGATIVE)
    AD: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    AI: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    UIP: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    UP: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    UT: float = parameter(default=40.0, domain=Domain.NONNEGATIVE)
    UD: float = parameter(default=40.0, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=1e-4, domain=Domain.NONNEGATIVE)
    b2: float = parameter(default=22.4, domain=Domain.POSITIVE)
    b3: float = parameter(default=22.4, domain=Domain.POSITIVE)
    c1: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    psi1_n: float = parameter(default=1.202e-4, domain=Domain.NONNEGATIVE)
    psi1_d: float = parameter(default=10.48e-4, domain=Domain.NONNEGATIVE)
    psi1_i: float = parameter(default=0.98e-4, domain=Domain.NONNEGATIVE)
    phi1_n: float = parameter(default=1.281e-4, domain=Domain.NONNEGATIVE)
    phi1_d: float = parameter(default=11.1e-4, domain=Domain.NONNEGATIVE)
    phi1_i: float = parameter(default=2.79e-4, domain=Domain.NONNEGATIVE)
    psi2: float = parameter(default=12.355, domain=Domain.POSITIVE)
    phi2: float = parameter(default=49.587, domain=Domain.POSITIVE)
    psi3_n: float = parameter(default=0.008, domain=Domain.NONNEGATIVE)
    psi3_d: float = parameter(default=0.150, domain=Domain.NONNEGATIVE)
    psi3_i: float = parameter(default=0.010, domain=Domain.NONNEGATIVE)
    psi4: float = parameter(default=62.162, domain=Domain.POSITIVE)
    Vc: float = parameter(default=5.56, domain=Domain.POSITIVE)
    v1: float = parameter(default=18.397, domain=Domain.NONNEGATIVE)
    v2: float = parameter(default=63.753, domain=Domain.POSITIVE)
    v3: float = parameter(default=28.4, domain=Domain.POSITIVE)
    phi: float = parameter(default=12.0, domain=Domain.POSITIVE)
    age: float = parameter(default=60.0, domain=Domain.POSITIVE)
    du1: float = parameter(default=1.515e-3, domain=Domain.NONNEGATIVE)
    du2: float = parameter(default=0.05635, domain=Domain.NONNEGATIVE)
    du3: float = parameter(default=5.776, domain=Domain.NONNEGATIVE)
    gamma_p: float = parameter(default=0.3125, domain=Domain.NONNEGATIVE)
    KP: float = parameter(default=800.0, domain=Domain.POSITIVE)
    beta_n: float = parameter(default=2.2622e-3, domain=Domain.NONNEGATIVE)
    beta_d: float = parameter(default=0.002, domain=Domain.NONNEGATIVE)
    beta_i: float = parameter(default=0.003, domain=Domain.NONNEGATIVE)
    sigma: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(
                s.N,
                _receptor_growth(s, s.psi1_n, s.phi1_n) * s.N * (1 - (s.N + s.c1 * cancer) / b1)
                - _receptor_death(s, s.psi3_n) * s.N,
            ),
            Rate(s.AD, (1 - s.m1) * division - _receptor_death(s, s.psi3_d) * s.AD),
            Rate(
                s.AI,
                _receptor_growth(s, s.psi1_i, s.phi1_i) * s.AI * (1 - (s.c1 * s.N + cancer) / s.b3)
                - _receptor_death(s, s.psi3_i) * s.AI
                + s.m1 * division,
            ),
            Rate(s.UIP, production - normal_leak - cancer_leak - s.du2 * s.UIP),
            Rate(s.UP, normal_leak + cancer_leak - s.du3 * s.UP),
        ]


class PiecewiseLinearProstate(Element):
    """Androgen-dependent, reversibly and irreversibly independent prostate cancer cells under intermittent therapy.

    During therapy, `u = 1`, the cells follow `dAD/dt = w11_on AD`, `dAIR/dt = w21_on AD + w22_on AIR`, and
    `dAII/dt = w31_on AD + w32_on AIR + w33_on AII`. Off therapy they follow `dAD/dt = w11_off AD + w12_off AIR`,
    `dAIR/dt = w22_off AIR`, and `dAII/dt = w33_off AII`. The therapy turns on when the PSA `x1 AD + x2 AIR + x3 AII`
    rises through `h1` and off when it falls through `h0`.

    Args:
        label:
            The label of the tumour.

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
        >>> from skcomplex.models.biology.oncology import PiecewiseLinearProstate
        >>> PiecewiseLinearProstate('tumour').h0
        4.0
    """

    AD: float = state(default=17.84, domain=Domain.NONNEGATIVE)
    AIR: float = state(default=0.778, domain=Domain.NONNEGATIVE)
    AII: float = state(default=0.382, domain=Domain.NONNEGATIVE)
    u: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    w11_on: float = parameter(default=-0.043844, domain=Domain.REAL)
    w21_on: float = parameter(default=0.00130716, domain=Domain.REAL)
    w22_on: float = parameter(default=-0.004282, domain=Domain.REAL)
    w31_on: float = parameter(default=0.0, domain=Domain.REAL)
    w32_on: float = parameter(default=0.0, domain=Domain.REAL)
    w33_on: float = parameter(default=0.00181, domain=Domain.REAL)
    w11_off: float = parameter(default=0.00223, domain=Domain.REAL)
    w12_off: float = parameter(default=0.1, domain=Domain.REAL)
    w22_off: float = parameter(default=0.00392, domain=Domain.REAL)
    w33_off: float = parameter(default=-0.2, domain=Domain.REAL)
    x1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    x2: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    x3: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    h0: float = parameter(default=4.0, domain=Domain.NONNEGATIVE)
    h1: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.AD, on * s.w11_on * s.AD + off * (s.w11_off * s.AD + s.w12_off * s.AIR)),
            Rate(s.AIR, on * (s.w21_on * s.AD + s.w22_on * s.AIR) + off * s.w22_off * s.AIR),
            Rate(s.AII, on * (s.w31_on * s.AD + s.w32_on * s.AIR + s.w33_on * s.AII) + off * s.w33_off * s.AII),
            Event(psa - s.h1, {s.u: 1}, direction=1, label='on'),
            Event(psa - s.h0, {s.u: 0}, direction=-1, label='off'),
        ]


class TamoxifenTumour(Element):
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
        >>> from skcomplex.models.biology.oncology import TamoxifenTumour
        >>> TamoxifenTumour('tumour').Asat
        10000.0
    """

    Ag: float = state(default=900.0, domain=Domain.NONNEGATIVE)
    As: float = state(default=50.0, domain=Domain.NONNEGATIVE)
    Am: float = state(default=50.0, domain=Domain.NONNEGATIVE)
    X0: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    X1: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    X2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    X3: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    E: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    mg: float = parameter(default=0.0013, domain=Domain.NONNEGATIVE)
    ms: float = parameter(default=0.039, domain=Domain.NONNEGATIVE)
    mm: float = parameter(default=0.0169, domain=Domain.NONNEGATIVE)
    Asat: float = parameter(default=1e4, domain=Domain.POSITIVE)
    a1: float = parameter(default=0.0062, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=0.010, domain=Domain.NONNEGATIVE)
    delta1: float = parameter(default=25.0, domain=Domain.NONNEGATIVE)
    delta2: float = parameter(default=25.0, domain=Domain.NONNEGATIVE)
    k01: float = parameter(default=0.048, domain=Domain.NONNEGATIVE)
    k12: float = parameter(default=0.993, domain=Domain.NONNEGATIVE)
    k23: float = parameter(default=35.932, domain=Domain.NONNEGATIVE)
    du1: float = parameter(default=1.145, domain=Domain.NONNEGATIVE)
    du2: float = parameter(default=39.525, domain=Domain.NONNEGATIVE)
    V: float = parameter(default=8.592, domain=Domain.POSITIVE)
    s: float = parameter(default=0.121, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.012, domain=Domain.NONNEGATIVE)
    u: float = parameter(default=12.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.Ag, -s.mg * s.Ag * room + 2 * s.mm * s.Am * room - s.a1 * s.Ag * (UX + s.delta1 * UH)),
            Rate(s.As, -s.ms * s.As + s.mg * s.Ag * room),
            Rate(s.Am, -s.mm * s.Am * room + s.ms * s.As),
            Rate(s.X0, -s.k01 * s.X0 + s.u),
            Rate(s.X1, s.k01 * s.X0 - s.k12 * s.X1),
            Rate(s.X2, s.k12 * s.X1 - (s.du1 + s.k23) * s.X2),
            Rate(s.X3, s.k23 * s.X2 - s.du2 * s.X3),
            Rate(s.E, s.s - s.d1 * s.E - s.a2 * s.E * (UX + s.delta2 * UH)),
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
    tumour = AndrogenSuppression(
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
    tumour = IntermittentSuppression(
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
    tumour = CellQuotaProstate(
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
    defaults={'until': 26280.0, 'dt': 168.0, 'solver': ScipyODE(method='LSODA')},
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
    prostate = AndrogenReceptorProstate('prostate', N=N, AD=AD, AI=AI, UT=UT, UD=UD, m1=m1, age=age)
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
    tumour = PiecewiseLinearProstate(
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
    defaults={'until': 2400.0, 'dt': 8.0, 'solver': ScipyODE(method='LSODA')},
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
    tumour = TamoxifenTumour('tumour', Ag=Ag, As=As, Am=Am, u=u, a1=a1, a2=a2)
    return System(elements=[tumour])
