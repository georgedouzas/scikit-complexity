"""Oncology models: tumour, immune, and normal cells under chemotherapy, with resistance, delays, and metastasis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System, delayed, where
from ....core.runtime import register_model

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


def _excess(U: Expression, threshold: Expression) -> Expression:
    """Return the drug concentration above its threshold, `(U - Uth) F(U - Uth)`, with `F` the Heaviside step."""
    return where(U - threshold >= 0, U - threshold, 0)


def _saturating(n: Expression, bd: Expression) -> Expression:
    """Return the Michaelis-Menten fraction `n / (bd + n)` of a population."""
    return n / (bd + n)


class TumourImmuneChemotherapy(Element):
    """Immune, tumour, and normal cells that compete and that a drug kills, after de Pillis and Radunskaya.

    The immune cells follow `dE/dt = s + rho E A / (alpha + A) - d1 E - c1 E A - a1 (1 - exp(-U)) E`, the tumour
    `dA/dt = r1 A (1 - b1 A) - c2 E A - c3 N A - a2 (1 - exp(-U)) A`, the normal cells
    `dN/dt = r2 N (1 - b2 N) - c4 N A - a3 (1 - exp(-U)) N`, and the drug `dU/dt = u - k U` under a constant infusion
    `u`.

    Args:
        label:
            The label of the patient.

        E:
            The initial immune cells.

        A:
            The initial tumour cells.

        N:
            The initial normal cells.

        U:
            The initial drug concentration.

        s:
            The influx of immune cells.

        rho:
            The immune response rate.

        alpha:
            The immune threshold.

        c1:
            The rate at which the tumour inactivates immune cells.

        c2:
            The rate at which immune cells kill the tumour.

        c3:
            The rate at which normal cells compete with the tumour.

        c4:
            The rate at which the tumour competes with normal cells.

        d1:
            The death rate of immune cells.

        r1:
            The growth rate of the tumour.

        r2:
            The growth rate of normal cells.

        b1:
            The reciprocal carrying capacity of the tumour.

        b2:
            The reciprocal carrying capacity of normal cells.

        a1:
            The fraction of immune cells the drug kills.

        a2:
            The fraction of tumour cells the drug kills.

        a3:
            The fraction of normal cells the drug kills.

        k:
            The decay rate of the drug, `dU` in the book.

        u:
            The constant infusion rate of the drug.

    Examples:
        >>> from skcomplex.models.biology.oncology import TumourImmuneChemotherapy
        >>> TumourImmuneChemotherapy('patient', u=1.0).u
        1.0
    """

    E: float = state(default=0.15, domain=Domain.NONNEGATIVE)
    A: float = state(default=0.25, domain=Domain.NONNEGATIVE)
    N: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    U: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    s: float = parameter(default=0.33, domain=Domain.NONNEGATIVE)
    rho: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.3, domain=Domain.POSITIVE)
    c1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    c3: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    c4: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=1.5, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    b2: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    k: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    u: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of the immune, tumour, and normal cells and of the drug.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, A, N, U = s.E, s.A, s.N, s.U
        kill = 1 - exp(-U)
        return [
            Rate(E, s.s + s.rho * E * A / (s.alpha + A) - s.d1 * E - s.c1 * E * A - s.a1 * kill * E),
            Rate(A, s.r1 * A * (1 - s.b1 * A) - s.c2 * E * A - s.c3 * N * A - s.a2 * kill * A),
            Rate(N, s.r2 * N * (1 - s.b2 * N) - s.c4 * N * A - s.a3 * kill * N),
            Rate(U, s.u - s.k * U),
        ]


class MultidrugResistance(Element):
    """Tumour cells split by their resistance to three drugs, each given by a constant infusion with its toxicity.

    The sensitive cells `S`, the cells resistant to one drug `R1`, `R2`, `R3`, to two drugs `R12`, `R13`, `R23`, and
    to all three `R123` grow at the same rate `r`. A fraction `m_i` of the offspring of a population gains resistance
    to drug `i`, so `dS/dt = r (1 - m1 - m2 - m3) S - K S`, `dRi/dt = r ((1 - mj - mk) Ri + mi S) - K Ri`, and so on,
    where `K` sums the kill `a_i (U_i - Uth) F(U_i - Uth)` of every drug the population is sensitive to. The drugs
    follow `dU_i/dt = u_i - k_i U_i` and their toxicities `dTX_i/dt = U_i - eta_i TX_i`.

    Args:
        label:
            The label of the tumour.

        S:
            The initial sensitive cells.

        R1:
            The initial cells resistant to drug 1.

        R2:
            The initial cells resistant to drug 2.

        R3:
            The initial cells resistant to drug 3.

        R12:
            The initial cells resistant to drugs 1 and 2.

        R13:
            The initial cells resistant to drugs 1 and 3.

        R23:
            The initial cells resistant to drugs 2 and 3.

        R123:
            The initial cells resistant to all drugs.

        U1:
            The initial concentration of drug 1.

        U2:
            The initial concentration of drug 2.

        U3:
            The initial concentration of drug 3.

        TX1:
            The initial toxicity of drug 1.

        TX2:
            The initial toxicity of drug 2.

        TX3:
            The initial toxicity of drug 3.

        r:
            The growth rate of every population.

        m1:
            The rate of mutation to resistance against drug 1.

        m2:
            The rate of mutation to resistance against drug 2.

        m3:
            The rate of mutation to resistance against drug 3.

        a1:
            The kill rate of drug 1 per unit of concentration above the threshold.

        a2:
            The kill rate of drug 2 per unit of concentration above the threshold.

        a3:
            The kill rate of drug 3 per unit of concentration above the threshold.

        Uth:
            The concentration above which a drug kills.

        k1:
            The decay rate of drug 1, `dU1` in the book.

        k2:
            The decay rate of drug 2, `dU2` in the book.

        k3:
            The decay rate of drug 3, `dU3` in the book.

        eta1:
            The metabolic rate that lowers the toxicity of drug 1.

        eta2:
            The metabolic rate that lowers the toxicity of drug 2.

        eta3:
            The metabolic rate that lowers the toxicity of drug 3.

        u1:
            The infusion rate of drug 1.

        u2:
            The infusion rate of drug 2.

        u3:
            The infusion rate of drug 3.

    Examples:
        >>> from skcomplex.models.biology.oncology import MultidrugResistance
        >>> MultidrugResistance('tumour').Uth
        10.0
    """

    S: float = state(default=4.605e11, domain=Domain.NONNEGATIVE)
    R1: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R3: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R12: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R13: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R23: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R123: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    U1: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    U2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    U3: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    TX1: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    TX2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    TX3: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r: float = parameter(default=0.0099, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=0.008, domain=Domain.NONNEGATIVE)
    m2: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    m3: float = parameter(default=0.014, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=0.0084, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=0.0074, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=0.0092, domain=Domain.NONNEGATIVE)
    Uth: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    k1: float = parameter(default=0.32, domain=Domain.NONNEGATIVE)
    k2: float = parameter(default=0.27, domain=Domain.NONNEGATIVE)
    k3: float = parameter(default=0.25, domain=Domain.NONNEGATIVE)
    eta1: float = parameter(default=0.4, domain=Domain.NONNEGATIVE)
    eta2: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    eta3: float = parameter(default=0.45, domain=Domain.NONNEGATIVE)
    u1: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    u2: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    u3: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth, mutation, and kill of the eight populations, and the drugs and their toxicities.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        kill1, kill2, kill3 = (a * _excess(U, s.Uth) for a, U in ((s.a1, s.U1), (s.a2, s.U2), (s.a3, s.U3)))
        m1, m2, m3, r = s.m1, s.m2, s.m3, s.r
        return [
            Rate(s.S, r * (1 - m1 - m2 - m3) * s.S - (kill1 + kill2 + kill3) * s.S),
            Rate(s.R1, r * ((1 - m2 - m3) * s.R1 + m1 * s.S) - (kill2 + kill3) * s.R1),
            Rate(s.R2, r * ((1 - m1 - m3) * s.R2 + m2 * s.S) - (kill1 + kill3) * s.R2),
            Rate(s.R3, r * ((1 - m1 - m2) * s.R3 + m3 * s.S) - (kill1 + kill2) * s.R3),
            Rate(s.R12, r * ((1 - m3) * s.R12 + m2 * s.R1 + m1 * s.R2) - kill3 * s.R12),
            Rate(s.R13, r * ((1 - m2) * s.R13 + m3 * s.R1 + m1 * s.R3) - kill2 * s.R13),
            Rate(s.R23, r * ((1 - m1) * s.R23 + m3 * s.R2 + m2 * s.R3) - kill1 * s.R23),
            Rate(s.R123, r * (s.R123 + m1 * s.R23 + m2 * s.R13 + m3 * s.R12)),
            Rate(s.U1, s.u1 - s.k1 * s.U1),
            Rate(s.U2, s.u2 - s.k2 * s.U2),
            Rate(s.U3, s.u3 - s.k3 * s.U3),
            Rate(s.TX1, s.U1 - s.eta1 * s.TX1),
            Rate(s.TX2, s.U2 - s.eta2 * s.TX2),
            Rate(s.TX3, s.U3 - s.eta3 * s.TX3),
        ]


class ResistanceSwitching(Element):
    """Sensitive cells and cells resistant by mutation or by a reversible phenotype switch, under a constant dose.

    With the total `V = S + Rm + Rp`, the sensitive cells follow
    `dS/dt = r1 (1 - b V) S - (m1 + m2) S - (alpha_m + alpha_p) u S - a u S + m5 Rp`, the mutated cells
    `dRm/dt = r2 (1 - b V) Rm + m1 S + alpha_m u S - am u Rm`, and the switched cells
    `dRp/dt = r3 (1 - b V) Rp + m2 S + alpha_p u S - m5 Rp - ap u Rp`. The rates `m1` and `m2` are the spontaneous
    (acquired) resistance, `alpha_m` and `alpha_p` the resistance the drug induces, and `m5` the resensitization of
    switched cells.

    Args:
        label:
            The label of the tumour.

        S:
            The initial sensitive cells.

        Rm:
            The initial cells resistant by mutation.

        Rp:
            The initial cells resistant by phenotype switching.

        r1:
            The growth rate of sensitive cells.

        r2:
            The growth rate of mutated cells.

        r3:
            The growth rate of switched cells.

        b:
            The reciprocal carrying capacity of the tumour.

        m1:
            The rate of spontaneous resistance by mutation.

        m2:
            The rate of spontaneous resistance by phenotype switching.

        alpha_m:
            The rate per unit of dose of resistance induced by mutation.

        alpha_p:
            The rate per unit of dose of resistance induced by phenotype switching.

        m5:
            The rate at which switched cells become sensitive again.

        a:
            The kill rate of sensitive cells per unit of dose.

        am:
            The kill rate of mutated cells per unit of dose.

        ap:
            The kill rate of switched cells per unit of dose.

        u:
            The effective dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import ResistanceSwitching
        >>> ResistanceSwitching('tumour').alpha_m
        0.01
    """

    S: float = state(default=0.01, domain=Domain.NONNEGATIVE)
    Rm: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Rp: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    r3: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=1e-6, domain=Domain.NONNEGATIVE)
    m2: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    alpha_m: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    alpha_p: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    m5: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    a: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    am: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    ap: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    u: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth, transitions, and kill of the sensitive, mutated, and switched cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        S, Rm, Rp, u = s.S, s.Rm, s.Rp, s.u
        room = 1 - s.b * (S + Rm + Rp)
        return [
            Rate(
                S,
                s.r1 * room * S - (s.m1 + s.m2) * S - (s.alpha_m + s.alpha_p) * u * S - s.a * u * S + s.m5 * Rp,
            ),
            Rate(Rm, s.r2 * room * Rm + s.m1 * S + s.alpha_m * u * S - s.am * u * Rm),
            Rate(Rp, s.r3 * room * Rp + s.m2 * S + s.alpha_p * u * S - s.m5 * Rp - s.ap * u * Rp),
        ]


class RestingHuntingChemotherapy(Element):
    """Hunting immune cells recruited from resting cells after a delay, a tumour, and a drug that kills all three.

    The hunting cells follow `dE/dt = m1 E Q(t - tau) - d1 E - c1 A E - a E / (bd + E) U`, the resting cells
    `dQ/dt = r1 Q (1 - b1 Q) - m1 E Q(t - tau) - a Q / (bd + Q) U`, the tumour
    `dA/dt = r2 A (1 - b2 A) - c2 A E - a A / (bd + A) U`, and the drug
    `dU/dt = u - (k + g (E / (bd + E) + Q / (bd + Q) + A / (bd + A))) U`, which binds to the cells it kills. Before
    the start the resting cells keep their initial value.

    Args:
        label:
            The label of the patient.

        E:
            The initial hunting cells.

        Q:
            The initial resting cells.

        A:
            The initial tumour cells.

        U:
            The initial drug concentration.

        m1:
            The rate at which resting cells become hunting cells.

        d1:
            The death rate of hunting cells.

        c1:
            The rate at which the tumour inactivates hunting cells.

        c2:
            The rate at which hunting cells kill the tumour.

        r1:
            The growth rate of resting cells.

        b1:
            The reciprocal carrying capacity of resting cells.

        r2:
            The growth rate of the tumour.

        b2:
            The reciprocal carrying capacity of the tumour.

        a:
            The kill rate of the drug on every population.

        bd:
            The half saturation of the kill.

        g:
            The rate at which the drug combines with every population.

        k:
            The elimination rate of the drug, `dU` in the book.

        tau:
            The delay of the conversion of resting cells to hunting cells.

        u:
            The constant infusion rate of the drug.

    Examples:
        >>> from skcomplex.models.biology.oncology import RestingHuntingChemotherapy
        >>> RestingHuntingChemotherapy('patient').tau
        45.6
    """

    E: float = state(default=0.01, domain=Domain.NONNEGATIVE)
    Q: float = state(default=0.48, domain=Domain.NONNEGATIVE)
    A: float = state(default=0.18, domain=Domain.NONNEGATIVE)
    U: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=0.093, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.0412, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=0.005133, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=1.6515, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.0245, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=1.5, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.18, domain=Domain.NONNEGATIVE)
    b2: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    a: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    bd: float = parameter(default=1e-4, domain=Domain.POSITIVE)
    g: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    k: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    tau: float = parameter(default=45.6, domain=Domain.POSITIVE)
    u: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the delayed recruitment of hunting cells, the growth of resting and tumour cells, and the drug.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, Q, A, U = s.E, s.Q, s.A, s.U
        fE, fQ, fA = _saturating(E, s.bd), _saturating(Q, s.bd), _saturating(A, s.bd)
        recruitment = s.m1 * E * delayed(Q, s.tau)
        return [
            Rate(E, recruitment - s.d1 * E - s.c1 * A * E - s.a * fE * U),
            Rate(Q, s.r1 * Q * (1 - s.b1 * Q) - recruitment - s.a * fQ * U),
            Rate(A, s.r2 * A * (1 - s.b2 * A) - s.c2 * A * E - s.a * fA * U),
            Rate(U, s.u - (s.k + s.g * (fE + fQ + fA)) * U),
        ]


class MetastasisChemotherapy(Element):
    """Normal and tumour cells at a primary and a secondary site, each with its drug, and a delayed migration.

    At each site the normal and tumour cells grow logistically, compete, and a drug kills them with a saturating
    response, `a N U / (bc + N)`. The drug at a site follows `dU/dt = u - (k + g1 N / (bc + N) + g2 A / (bc + A)) U`.
    Tumour cells leave the primary site at the rate `delta`, and a fraction `epsilon` of them reaches the secondary site
    after the delay `tau`, `dAs/dt = ... + epsilon delta Ap(t - tau)`. Before the start the primary tumour keeps its
    initial value.

    Args:
        label:
            The label of the patient.

        Np:
            The initial normal cells at the primary site.

        Ap:
            The initial tumour cells at the primary site.

        Up:
            The initial drug concentration at the primary site.

        Ns:
            The initial normal cells at the secondary site.

        As:
            The initial tumour cells at the secondary site.

        Us:
            The initial drug concentration at the secondary site.

        r1:
            The growth rate of normal cells at the primary site.

        r2:
            The growth rate of tumour cells at the primary site.

        r3:
            The growth rate of normal cells at the secondary site.

        r4:
            The growth rate of tumour cells at the secondary site.

        b1:
            The reciprocal carrying capacity of normal cells at the primary site.

        b2:
            The reciprocal carrying capacity of tumour cells at the primary site.

        b3:
            The reciprocal carrying capacity of normal cells at the secondary site.

        b4:
            The reciprocal carrying capacity of tumour cells at the secondary site.

        c1:
            The rate at which the tumour competes with normal cells at the primary site.

        c2:
            The rate at which normal cells compete with the tumour at the primary site.

        c3:
            The rate at which the tumour competes with normal cells at the secondary site.

        c4:
            The rate at which normal cells compete with the tumour at the secondary site.

        a1:
            The kill rate of the drug on normal cells at the primary site.

        a2:
            The kill rate of the drug on tumour cells at the primary site.

        a3:
            The kill rate of the drug on normal cells at the secondary site.

        a4:
            The kill rate of the drug on tumour cells at the secondary site.

        g1:
            The rate at which the drug combines with normal cells at the primary site.

        g2:
            The rate at which the drug combines with tumour cells at the primary site.

        g3:
            The rate at which the drug combines with normal cells at the secondary site.

        g4:
            The rate at which the drug combines with tumour cells at the secondary site.

        bc:
            The half saturation of the kill and of the binding.

        k:
            The elimination rate of the drug at both sites, `dU1` and `dU2` in the book.

        delta:
            The rate at which tumour cells leave the primary site.

        epsilon:
            The fraction of leaving cells that reaches the secondary site.

        tau:
            The delay of the migration.

        up:
            The infusion rate of the drug at the primary site.

        us:
            The infusion rate of the drug at the secondary site.

    Examples:
        >>> from skcomplex.models.biology.oncology import MetastasisChemotherapy
        >>> MetastasisChemotherapy('patient').epsilon
        0.0001
    """

    Np: float = state(default=1460.0, domain=Domain.NONNEGATIVE)
    Ap: float = state(default=165.0, domain=Domain.NONNEGATIVE)
    Up: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Ns: float = state(default=1750.0, domain=Domain.NONNEGATIVE)
    As: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Us: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=1.5, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    r3: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    r4: float = parameter(default=12.0, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=0.00068, domain=Domain.NONNEGATIVE)
    b2: float = parameter(default=0.00047, domain=Domain.NONNEGATIVE)
    b3: float = parameter(default=0.00057, domain=Domain.NONNEGATIVE)
    b4: float = parameter(default=0.00043, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=0.0075, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=0.005, domain=Domain.NONNEGATIVE)
    c3: float = parameter(default=0.002, domain=Domain.NONNEGATIVE)
    c4: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=0.0008, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=0.08, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=0.0006, domain=Domain.NONNEGATIVE)
    a4: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    g1: float = parameter(default=0.0024, domain=Domain.NONNEGATIVE)
    g2: float = parameter(default=0.24, domain=Domain.NONNEGATIVE)
    g3: float = parameter(default=0.0018, domain=Domain.NONNEGATIVE)
    g4: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    bc: float = parameter(default=1.0, domain=Domain.POSITIVE)
    k: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)
    delta: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    epsilon: float = parameter(default=0.0001, domain=Domain.NONNEGATIVE)
    tau: float = parameter(default=1.0, domain=Domain.POSITIVE)
    up: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    us: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the cells and drugs at both sites and the delayed arrival of tumour cells at the secondary site.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        Np, Ap, Up, Ns, As, Us = s.Np, s.Ap, s.Up, s.Ns, s.As, s.Us
        fNp, fAp, fNs, fAs = (_saturating(n, s.bc) for n in (Np, Ap, Ns, As))
        arrival = s.epsilon * s.delta * delayed(Ap, s.tau)
        return [
            Rate(Np, s.r1 * Np * (1 - s.b1 * Np) - s.c1 * Np * Ap - s.a1 * fNp * Up),
            Rate(Ap, s.r2 * Ap * (1 - s.b2 * Ap) - s.c2 * Np * Ap - s.a2 * fAp * Up - s.delta * Ap),
            Rate(Up, s.up - (s.k + s.g1 * fNp + s.g2 * fAp) * Up),
            Rate(Ns, s.r3 * Ns * (1 - s.b3 * Ns) - s.c3 * Ns * As - s.a3 * fNs * Us),
            Rate(As, s.r4 * As * (1 - s.b4 * As) - s.c4 * Ns * As - s.a4 * fAs * Us + arrival),
            Rate(Us, s.us - (s.k + s.g3 * fNs + s.g4 * fAs) * Us),
        ]


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'tumour-immune', 'competition'),
    references=(BOOK, 'de Pillis and Radunskaya 2003'),
    defaults={'until': 150.0, 'dt': 0.5},
)
def de_pillis_radunskaya(
    E: float = 0.15,
    A: float = 0.25,
    N: float = 1.0,
    *,
    u: float = 0.0,
    s: float = 0.33,
    r1: float = 1.5,
    a2: float = 0.3,
) -> System:
    """Return the de Pillis-Radunskaya model of immune, tumour, and normal cells under a constant drug infusion.

    The cells are normalised so that the carrying capacity of normal cells is 1, and time is in days. Untreated, a
    tumour that starts at 0.25 escapes the immune cells and settles at the high equilibrium where it outnumbers the
    normal cells. A constant infusion `u = 1` clears it, while the normal cells stay above 0.7. The parameters are those
    of Table 3.3 of Padmanabhan et al. The book's initial tumour of `1e-5` lies in the basin of the tumour-free
    equilibrium, which is stable since `r1 < c2 s / d1 + c3`, so the defaults are the initial conditions of de Pillis
    and Radunskaya 2003, `E = 0.15`, `A = 0.25`, `N = 1`. The book prints no infusion schedule, so the infusion is a
    constant rate `u` per day, untreated by default.

    Args:
        E:
            The initial immune cells.

        A:
            The initial tumour cells.

        N:
            The initial normal cells.

        u:
            The constant infusion rate of the drug.

        s:
            The influx of immune cells.

        r1:
            The growth rate of the tumour.

        a2:
            The fraction of tumour cells the drug kills.

    Returns:
        system:
            The model, with the cells `patient.E`, `patient.A`, `patient.N` and the drug `patient.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import de_pillis_radunskaya
        >>> de_pillis_radunskaya(u=1.0).elements[0].u
        1.0
    """
    return System(components=[TumourImmuneChemotherapy('patient', E=E, A=A, N=N, u=u, s=s, r1=r1, a2=a2)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'drug resistance', 'toxicity'),
    references=(BOOK, 'Tse et al. 2007'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def multidrug_resistance(
    S: float = 4.605e11,
    *,
    r: float = 0.0099,
    u1: float = 10.0,
    u2: float = 10.0,
    u3: float = 10.0,
) -> System:
    """Return a tumour of eight populations resistant to subsets of three drugs, given by constant infusions.

    Each drug kills the populations sensitive to it once its concentration exceeds a threshold, so the drugs clear the
    sensitive cells while the cells resistant to all three keep growing. Time is in days, the cells are counted, and the
    drugs are in dosage units D. The parameters are those of Table 3.4 of Padmanabhan et al. The book prints the
    triple-resistant population without its own growth, `dR123/dt = r (m1 R23 + m2 R13 + m3 R12)`, against its
    assumption that all populations grow at the same rate, so the model adds the growth `r R123`. The book prints no
    infusion schedule, so the drugs are constant infusions whose concentrations settle within the book's range of 0 to
    50.

    Args:
        S:
            The initial sensitive cells, all cells resistant to any drug starting at 0.

        r:
            The growth rate of every population.

        u1:
            The infusion rate of drug 1.

        u2:
            The infusion rate of drug 2.

        u3:
            The infusion rate of drug 3.

    Returns:
        system:
            The model, with the populations `tumour.S`, `tumour.R1` to `tumour.R123`, the drugs `tumour.U1` to
            `tumour.U3`, and their toxicities `tumour.TX1` to `tumour.TX3`.

    Examples:
        >>> from skcomplex.models.biology.oncology import multidrug_resistance
        >>> multidrug_resistance(u2=0.0).elements[0].u2
        0.0
    """
    return System(components=[MultidrugResistance('tumour', S=S, r=r, u1=u1, u2=u2, u3=u3)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'drug resistance', 'phenotype switching'),
    references=(BOOK, 'Greene, Sanchez-Tapia, and Sontag 2018', 'Greene, Gevertz, and Sontag 2019'),
    defaults={'until': 400.0, 'dt': 1.0},
)
def drug_resistance_switching(
    S: float = 0.01,
    *,
    u: float = 1.0,
    m1: float = 1e-6,
    m2: float = 0.0,
    alpha_m: float = 0.01,
    alpha_p: float = 0.0,
    m5: float = 0.0,
    r2: float = 0.2,
    r3: float = 0.2,
) -> System:
    """Return sensitive cells and cells resistant by mutation or by a reversible phenotype switch under a dose.

    The drug kills the sensitive cells, so the resistant cells, which grow slower, take the tumour over. Resistance is
    acquired spontaneously at the rates `m1` and `m2` and induced by the drug at the rates `alpha_m` and `alpha_p` per
    unit of dose, and switched cells return to sensitivity at the rate `m5`. Time and the cells are dimensionless, the
    carrying capacity being 1. The book gives values only for the two-population model of Greene, Sanchez-Tapia, and
    Sontag, its Table 3.5, which is this model without switched cells. The defaults are that model, with the
    induced rate `alpha_m = 0.01` of the table and an effective dose of 1, so switching is off until `m2`, `alpha_p`,
    or `m5` is set.

    Args:
        S:
            The initial sensitive cells, the resistant cells starting at 0.

        u:
            The effective dose.

        m1:
            The rate of spontaneous resistance by mutation.

        m2:
            The rate of spontaneous resistance by phenotype switching.

        alpha_m:
            The rate per unit of dose of resistance induced by mutation.

        alpha_p:
            The rate per unit of dose of resistance induced by phenotype switching.

        m5:
            The rate at which switched cells become sensitive again.

        r2:
            The growth rate of mutated cells relative to sensitive cells.

        r3:
            The growth rate of switched cells relative to sensitive cells.

    Returns:
        system:
            The model, with the cells `tumour.S`, `tumour.Rm`, and `tumour.Rp`.

    Examples:
        >>> from skcomplex.models.biology.oncology import drug_resistance_switching
        >>> drug_resistance_switching(alpha_m=0.0).elements[0].alpha_m
        0.0
    """
    tumour = ResistanceSwitching(
        'tumour',
        S=S,
        u=u,
        m1=m1,
        m2=m2,
        alpha_m=alpha_m,
        alpha_p=alpha_p,
        m5=m5,
        r2=r2,
        r3=r3,
    )
    return System(components=[tumour])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'dde', 'delay', 'tumour-immune'),
    references=(BOOK, 'Banerjee and Sarkar 2008', 'Borges et al. 2014'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def delayed_resting_hunting_chemotherapy(
    E: float = 0.01,
    Q: float = 0.48,
    A: float = 0.18,
    *,
    u: float = 0.0,
    tau: float = 45.6,
    a: float = 0.001,
) -> System:
    """Return the tumour-immune model of resting and hunting cells with a conversion delay, under a drug infusion.

    Resting immune cells become hunting cells after the delay `tau`, and hunting cells kill the tumour. Time and the
    cells are dimensionless. The parameters and initial conditions are those of Table 3.6 of Padmanabhan et al., and
    the infusion is a constant rate `u`, untreated by default, within the book's range of 0 to 0.03.

    Args:
        E:
            The initial hunting cells.

        Q:
            The initial resting cells, also their history.

        A:
            The initial tumour cells.

        u:
            The constant infusion rate of the drug.

        tau:
            The delay of the conversion of resting cells to hunting cells.

        a:
            The kill rate of the drug on every population.

    Returns:
        system:
            The model, with the cells `patient.E`, `patient.Q`, `patient.A` and the drug `patient.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import delayed_resting_hunting_chemotherapy
        >>> delayed_resting_hunting_chemotherapy(tau=10.0).elements[0].tau
        10.0
    """
    return System(components=[RestingHuntingChemotherapy('patient', E=E, Q=Q, A=A, u=u, tau=tau, a=a)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'dde', 'delay', 'metastasis'),
    references=(BOOK, 'Pinho, Freedman, and Nani 2002'),
    defaults={'until': 10.0, 'dt': 0.05},
)
def delayed_metastasis(
    Np: float = 1460.0,
    Ap: float = 165.0,
    Ns: float = 1750.0,
    As: float = 0.0,
    *,
    up: float = 0.0,
    us: float = 0.0,
    tau: float = 1.0,
) -> System:
    """Return normal and tumour cells at a primary and a secondary site, with a delayed migration of tumour cells.

    Untreated, the faster tumour cells outcompete the normal cells at the primary site, and the cells that migrate seed
    a tumour that does the same at the secondary site. The parameters are those of Table 3.7 of Padmanabhan et al.,
    with time in days, and the initial cells are at the upper ends of the book's ranges with the secondary site free
    of tumour. The book prints the kill at the secondary site with the infusion `us` in place of the concentration
    `Us`, so the model uses `Us` as at the primary site. The infusions are constant rates, untreated by default, within
    the book's range of 0 to 650.

    Args:
        Np:
            The initial normal cells at the primary site.

        Ap:
            The initial tumour cells at the primary site, also their history.

        Ns:
            The initial normal cells at the secondary site.

        As:
            The initial tumour cells at the secondary site.

        up:
            The infusion rate of the drug at the primary site.

        us:
            The infusion rate of the drug at the secondary site.

        tau:
            The delay of the migration.

    Returns:
        system:
            The model, with the cells `patient.Np`, `patient.Ap`, `patient.Ns`, `patient.As` and the drugs
            `patient.Up` and `patient.Us`.

    Examples:
        >>> from skcomplex.models.biology.oncology import delayed_metastasis
        >>> delayed_metastasis(up=650.0).elements[0].up
        650.0
    """
    return System(components=[MetastasisChemotherapy('patient', Np=Np, Ap=Ap, Ns=Ns, As=As, up=up, us=us, tau=tau)])
