"""Oncology models: cell-cycle compartments and red-cell production under chemotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import delayed
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, StateSpec, VisualSpec

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class CellCycleElement(Element):
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
        >>> from skcomplex.models.biology.oncology import CellCycleElement
        >>> CellCycleElement('tumour').l2
        0.395
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.8,
        A3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        l1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.197,
        l2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.395,
        l3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.107,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.u1 = u1
        self.u2 = u2

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(s.A1, -s.l1 * s.A1 + 2 * (1 - s.u1) * s.l3 * s.A3),
            RateLaw(s.A2, s.l1 * s.A1 - exit_s),
            RateLaw(s.A3, exit_s - s.l3 * s.A3),
        ]


class DormantCellCycleElement(Element):
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
        >>> from skcomplex.models.biology.oncology import DormantCellCycleElement
        >>> DormantCellCycleElement('tumour').b0
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A0: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 6.5,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        l0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        l1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        l2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        v: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.b0 = b0
        self.b1 = b1
        self.u = u
        self.v = v

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(s.A0, -recruited + s.b0 * daughters),
            RateLaw(s.A1, recruited - s.l1 * s.A1 + s.b1 * daughters),
            RateLaw(s.A2, s.l1 * s.A1 - s.l2 * s.A2),
        ]


class ErythropoiesisElement(Element):
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
        >>> from skcomplex.models.biology.oncology import ErythropoiesisElement
        >>> ErythropoiesisElement('patient').lifespan
        120.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        L: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.84,
        Q: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 19.6,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2.16,
        MCV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 85.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.00139,
        d1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0071,
        a0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00022,
        tau_h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.68,
        MCV_h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 85.0,
        MCV0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 85.0,
        g1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0317,
        m0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.01,
        Q50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.027,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lifespan: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 120.0,
        U: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.L = L
        self.Q = Q
        self.P = P
        self.MCV = MCV
        self.r1 = r1
        self.d1 = d1
        self.a0 = a0
        self.tau_h = tau_h
        self.MCV_h = MCV_h
        self.MCV0 = MCV0
        self.g1 = g1
        self.m0 = m0
        self.Q50 = Q50
        self.n = n
        self.lifespan = lifespan
        self.U = U

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(Q, -s.r1 * Q + (2 * survival - 1) * entry),
            RateLaw(P, -a1 * P + (1 - survival) * entry),
            RateLaw(L, s.r1 * Q - s.d1 * old_L),
            RateLaw(MCV, (new_volume * s.r1 * Q - 0.85 * old_MCV * s.d1 * old_L - 0.15 * MCV * s.d1 * L) / L),
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
        'DormantCellCycleElement'
    """
    if dormant:
        return System(elements=[DormantCellCycleElement('tumour', u=u1, v=v)])
    return System(elements=[CellCycleElement('tumour', u1=u1, u2=u2)])


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
    patient = ErythropoiesisElement(
        'patient',
        L=r1 * Q / d1,
        Q=Q,
        P=0.71 / 6.43 * Q,
        r1=r1,
        d1=d1,
        m0=m0,
        Q50=Q50,
        n=n,
        U=U,
    )
    return System(elements=[patient])
