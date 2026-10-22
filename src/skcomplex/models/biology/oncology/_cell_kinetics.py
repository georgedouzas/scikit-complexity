"""Oncology models: cell-cycle compartments and red-cell production under chemotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System, delayed
from ....core.runtime import register_model

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class CellCycle(Element):
    """Tumour cells in three phases of the cell cycle under a killing and a blocking agent.

    The compartments hold the cells in G0 and G1 `A1`, in S `A2`, and in G2 and M `A3`, and follow
    `dA1/dt = -l1 A1 + 2 (1 - u1) l3 A3`, `dA2/dt = l1 A1 - (1 - u2) l2 A2`, and `dA3/dt = (1 - u2) l2 A2 - l3 A3`.
    The killing agent kills the fraction `u1` of the cells that divide, and the blocking agent slows the transit
    through S by the fraction `u2`.

    Args:
        label:
            The label of the tumour.

        A1:
            The initial cells in G0 and G1.

        A2:
            The initial cells in S.

        A3:
            The initial cells in G2 and M.

        l1:
            The rate of exit from G0 and G1.

        l2:
            The rate of exit from S.

        l3:
            The rate of exit from G2 and M.

        u1:
            The fraction of dividing cells the killing agent kills.

        u2:
            The fraction by which the blocking agent slows the transit through S.

    Examples:
        >>> from skcomplex.models.biology.oncology import CellCycle
        >>> CellCycle('tumour').l2
        0.395
    """

    A1: float = state(default=15.0, domain=Domain.NONNEGATIVE)
    A2: float = state(default=3.8, domain=Domain.NONNEGATIVE)
    A3: float = state(default=0.8, domain=Domain.NONNEGATIVE)
    l1: float = parameter(default=0.197, domain=Domain.POSITIVE)
    l2: float = parameter(default=0.395, domain=Domain.POSITIVE)
    l3: float = parameter(default=0.107, domain=Domain.POSITIVE)
    u1: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    u2: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the transits between the three compartments and the division with its kill.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        exit_s = (1 - s.u2) * s.l2 * s.A2
        return [
            Rate(s.A1, -s.l1 * s.A1 + 2 * (1 - s.u1) * s.l3 * s.A3),
            Rate(s.A2, s.l1 * s.A1 - exit_s),
            Rate(s.A3, exit_s - s.l3 * s.A3),
        ]


class DormantCellCycle(Element):
    """Tumour cells in a dormant phase and two cycling phases under a killing and a recruiting agent.

    The compartments hold the dormant cells in G0 `A0`, the cells in G1 and S `A1`, and the cells in G2 and M `A2`.
    A fraction `b0` of the daughters enters G0 and `b1` enters G1, so `dA0/dt = -l0 (1 + v) A0 + 2 b0 (1 - u) l2 A2`,
    `dA1/dt = l0 (1 + v) A0 - l1 A1 + 2 b1 (1 - u) l2 A2`, and `dA2/dt = l1 A1 - l2 A2`. The killing agent kills the
    fraction `u` of the cells that divide, and the recruiting agent speeds the exit from G0 by the factor `1 + v`.

    Args:
        label:
            The label of the tumour.

        A0:
            The initial dormant cells.

        A1:
            The initial cells in G1 and S.

        A2:
            The initial cells in G2 and M.

        l0:
            The rate of exit from G0.

        l1:
            The rate of exit from G1 and S.

        l2:
            The rate of exit from G2 and M.

        b0:
            The fraction of daughters that enters G0.

        b1:
            The fraction of daughters that enters G1.

        u:
            The fraction of dividing cells the killing agent kills.

        v:
            The relative increase of the exit from G0 the recruiting agent causes.

    Examples:
        >>> from skcomplex.models.biology.oncology import DormantCellCycle
        >>> DormantCellCycle('tumour').b0
        0.9
    """

    A0: float = state(default=6.5, domain=Domain.NONNEGATIVE)
    A1: float = state(default=0.2, domain=Domain.NONNEGATIVE)
    A2: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    l0: float = parameter(default=0.05, domain=Domain.POSITIVE)
    l1: float = parameter(default=0.5, domain=Domain.POSITIVE)
    l2: float = parameter(default=1.0, domain=Domain.POSITIVE)
    b0: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    u: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    v: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the recruitment from G0, the transits, and the division with its kill.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        recruited = s.l0 * (1 + s.v) * s.A0
        daughters = 2 * (1 - s.u) * s.l2 * s.A2
        return [
            Rate(s.A0, -recruited + s.b0 * daughters),
            Rate(s.A1, recruited - s.l1 * s.A1 + s.b1 * daughters),
            Rate(s.A2, s.l1 * s.A1 - s.l2 * s.A2),
        ]


class Erythropoiesis(Element):
    """Quiescent, proliferating, and mature red cells and their mean volume under a constant 6-mercaptopurine level.

    The drug sets the death rate of proliferating cells `a1 = a0 U` and the cell-cycle time
    `tau = (g1 U + MCV0) tau_h / MCV_h`. Quiescent cells enter the cycle at the rate
    `m(Q) = m0 Q50^n / (Q50^n + Q^n)`, of which the fraction `exp(-a1 tau)` survives to give two daughters, and leave
    to mature at the rate `r1`, so `dQ/dt = -r1 Q + (2 exp(-a1 tau) - 1) m(Q) Q`. The proliferating cells follow
    `dP/dt = -a1 P + (1 - exp(-a1 tau)) m(Q) Q`, the mature cells, which live `lifespan` days,
    `dL/dt = r1 Q - d1 L(t - lifespan)`, and their mean corpuscular volume
    `dMCV/dt = ((g1 U + MCV0) r1 Q - 0.85 MCV(t - lifespan) d1 L(t - lifespan) - 0.15 MCV d1 L) / L`. Before the start
    the mature cells and their volume keep their initial values.

    Args:
        label:
            The label of the patient.

        L:
            The initial mature red cells.

        Q:
            The initial quiescent cells.

        P:
            The initial proliferating cells.

        MCV:
            The initial mean corpuscular volume.

        r1:
            The rate of differentiation of quiescent cells.

        d1:
            The death rate of mature cells.

        a0:
            The death rate of proliferating cells per unit of drug.

        tau_h:
            The cell-cycle time of healthy cells.

        MCV_h:
            The healthy mean corpuscular volume.

        MCV0:
            The volume of new cells without the drug.

        g1:
            The increase of the volume of new cells per unit of drug.

        m0:
            The maximum rate at which quiescent cells enter the cycle.

        Q50:
            The quiescent cells at which the rate of entry is half its maximum.

        n:
            The sensitivity of the rate of entry to the quiescent cells.

        lifespan:
            The life span of mature cells.

        U:
            The concentration of the drug.

    Examples:
        >>> from skcomplex.models.biology.oncology import Erythropoiesis
        >>> Erythropoiesis('patient').lifespan
        120.0
    """

    L: float = state(default=3.84, domain=Domain.NONNEGATIVE)
    Q: float = state(default=19.6, domain=Domain.NONNEGATIVE)
    P: float = state(default=2.16, domain=Domain.NONNEGATIVE)
    MCV: float = state(default=85.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.00139, domain=Domain.POSITIVE)
    d1: float = parameter(default=0.0071, domain=Domain.POSITIVE)
    a0: float = parameter(default=0.00022, domain=Domain.NONNEGATIVE)
    tau_h: float = parameter(default=0.68, domain=Domain.POSITIVE)
    MCV_h: float = parameter(default=85.0, domain=Domain.POSITIVE)
    MCV0: float = parameter(default=85.0, domain=Domain.POSITIVE)
    g1: float = parameter(default=0.0317, domain=Domain.NONNEGATIVE)
    m0: float = parameter(default=1.01, domain=Domain.POSITIVE)
    Q50: float = parameter(default=0.027, domain=Domain.POSITIVE)
    n: float = parameter(default=1.0, domain=Domain.POSITIVE)
    lifespan: float = parameter(default=120.0, domain=Domain.POSITIVE)
    U: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of quiescent, proliferating, and mature cells and of their mean volume.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        L, Q, P, MCV = s.L, s.Q, s.P, s.MCV
        a1 = s.a0 * s.U
        new_volume = s.g1 * s.U + s.MCV0
        survival = exp(-a1 * new_volume * s.tau_h / s.MCV_h)
        entry = s.m0 * s.Q50**s.n / (s.Q50**s.n + Q**s.n) * Q
        old_L = delayed(L, s.lifespan)
        old_MCV = delayed(MCV, s.lifespan)
        return [
            Rate(Q, -s.r1 * Q + (2 * survival - 1) * entry),
            Rate(P, -a1 * P + (1 - survival) * entry),
            Rate(L, s.r1 * Q - s.d1 * old_L),
            Rate(MCV, (new_volume * s.r1 * Q - 0.85 * old_MCV * s.d1 * old_L - 0.15 * MCV * s.d1 * L) / L),
        ]


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'cell cycle', 'compartments'),
    references=(BOOK, 'Świerniak, Ledzewicz, and Schättler 2003'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def cell_cycle_chemotherapy(
    *,
    dormant: bool = False,
    u1: float = 0.0,
    u2: float = 0.0,
    v: float = 0.0,
) -> System:
    """Return tumour cells in cell-cycle compartments under killing, blocking, and recruiting agents.

    Without a dormant phase the cells cycle through G0 and G1, S, and G2 and M, a killing agent kills a fraction `u1` of
    the dividing cells, and a blocking agent slows the transit through S by `u2`. With a dormant phase the cells rest
    in G0, cycle through G1 and S and through G2 and M, the killing agent kills `u1` of the dividing cells, and a
    recruiting agent speeds the exit from G0 by the factor `1 + v`. Each division gives `2 (1 - u1)` cells, so the
    tumour grows for `u1 < 1/2` and shrinks for `u1 > 1/2`, whatever the blocking and the recruitment. The parameters
    and initial cells are those of Tables 3.8 and 3.9 of Padmanabhan et al., in cell counts and per unit time. The
    book prints the agents as additive terms `2 l3 u1` and garbles the indices of the dormant model, so the model
    takes the form of Świerniak et al., where the agents act on the cells of the compartment they leave. Constant
    agents are given, untreated by default.

    Args:
        dormant:
            Whether the cells have a dormant phase, the model of Table 3.9, or not, the model of Table 3.8.

        u1:
            The fraction of dividing cells the killing agent kills.

        u2:
            The fraction by which the blocking agent slows the transit through S, without a dormant phase.

        v:
            The relative increase of the exit from G0 the recruiting agent causes, with a dormant phase.

    Returns:
        system:
            The model, with the cells `tumour.A1` to `tumour.A3`, or `tumour.A0` to `tumour.A2` with a dormant phase.

    Examples:
        >>> from skcomplex.models.biology.oncology import cell_cycle_chemotherapy
        >>> type(cell_cycle_chemotherapy(dormant=True).elements[0]).__name__
        'DormantCellCycle'
    """
    if dormant:
        return System(components=[DormantCellCycle('tumour', u=u1, v=v)])
    return System(components=[CellCycle('tumour', u1=u1, u2=u2)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'dde', 'delay', 'haematopoiesis', 'leukaemia'),
    references=(BOOK, 'Noble et al. 2010', 'Mackey 1978'),
    defaults={'until': 3000.0, 'dt': 10.0},
)
def erythropoiesis_mercaptopurine(
    *,
    U: float = 500.0,
    r1: float = 0.00139,
    d1: float = 0.0071,
    m0: float = 1.01,
    Q50: float = 0.027,
    n: float = 1.0,
) -> System:
    """Return the production of red cells under a constant level of 6-mercaptopurine, starting from health.

    The drug kills proliferating cells and lengthens their cycle, so fewer quiescent cells survive to mature, the mature
    red cells fall, and their mean volume rises to `MCV0 + g1 U`. The cells are in 10^12 per kg, the volume in fL, the
    drug in pMol per 8 x 10^8 red cells, and time in days. The parameters are those of Table 3.10 of Padmanabhan et al.,
    with its half saturation of 2.7 x 10^10 cells per kg written as `Q50 = 0.027`. The patient starts at the healthy
    steady state, `Q = Q50 (m0 / r1 - 1)^(1/n)`, `L = r1 Q / d1`, the book's ratio of proliferating to quiescent cells
    0.71 / 6.43, and `MCV = 85`. The book garbles equation 3.36 as `-r1 AQ (2 exp(a1 tau) - 1) m(AQ) AQ`. The model
    takes the standard form of Mackey 1978 that Noble et al. use, `dQ/dt = -r1 Q + (2 exp(-a1 tau) - 1) m(Q) Q`, in
    which cells leave to mature at the rate `r1` and the cells that enter the cycle return doubled once they survive the
    drug. Its healthy steady state gives 3.8 x 10^12 mature cells per kg, within the book's range.

    Args:
        U:
            The concentration of the drug, which the book ranges from 158 to 1000.

        r1:
            The rate of differentiation of quiescent cells.

        d1:
            The death rate of mature cells.

        m0:
            The maximum rate at which quiescent cells enter the cycle.

        Q50:
            The quiescent cells at which the rate of entry is half its maximum.

        n:
            The sensitivity of the rate of entry to the quiescent cells.

    Returns:
        system:
            The model, with the cells `patient.Q`, `patient.P`, `patient.L` and the mean volume `patient.MCV`.

    Examples:
        >>> from skcomplex.models.biology.oncology import erythropoiesis_mercaptopurine
        >>> erythropoiesis_mercaptopurine(U=0.0).elements[0].MCV
        85.0
    """
    Q = Q50 * (m0 / r1 - 1) ** (1 / n)
    patient = Erythropoiesis('patient', L=r1 * Q / d1, Q=Q, P=0.71 / 6.43 * Q, r1=r1, d1=d1, m0=m0, Q50=Q50, n=n, U=U)
    return System(components=[patient])
