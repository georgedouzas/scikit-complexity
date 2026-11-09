"""Cell-cycle compartments and red-cell production under chemotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws.utils import delayed
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class CellCycleElement(Element):
    """Tumour cells in three phases of the cell cycle.

    The cells are under a killing and a blocking agent.

    The compartments hold the cells in G0 and G1 `A1`, in S `A2`, and in G2 and M `A3`, and follow
    `dA1/dt = -l1 A1 + 2 (1 - u1) l3 A3`, `dA2/dt = l1 A1 - (1 - u2) l2 A2`, and `dA3/dt = (1 - u2) l2 A2 - l3 A3`.
    The killing agent kills the fraction `u1` of the cells that divide, and the blocking agent slows the transit
    through S by the fraction `u2`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `CellCycleChemotherapySystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.oncology import CellCycleElement
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


class DormantCellCycleElement(Element):
    """Tumour cells in a dormant phase and two cycling phases.

    The cells are under a killing and a recruiting agent.

    The compartments hold the dormant cells in G0 `A0`, the cells in G1 and S `A1`, and the cells in G2 and M `A2`.
    A fraction `b0` of the daughters enters G0 and `b1` enters G1, so `dA0/dt = -l0 (1 + v) A0 + 2 b0 (1 - u) l2 A2`,
    `dA1/dt = l0 (1 + v) A0 - l1 A1 + 2 b1 (1 - u) l2 A2`, and `dA2/dt = l1 A1 - l2 A2`. The killing agent kills the
    fraction `u` of the cells that divide, and the recruiting agent speeds the exit from G0 by the factor `1 + v`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `CellCycleChemotherapySystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.oncology import DormantCellCycleElement
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


class ErythropoiesisElement(Element):
    """Quiescent, proliferating, and mature red cells and their mean volume.

    The cells are under a constant 6-mercaptopurine level.

    The drug sets the death rate of proliferating cells `a1 = a0 U` and the cell-cycle time
    `tau = (g1 U + MCV0) tau_h / MCV_h`. Quiescent cells enter the cycle at the rate
    `m(Q) = m0 Q50^n / (Q50^n + Q^n)`, of which the fraction `exp(-a1 tau)` survives to give two daughters, and leave
    to mature at the rate `r1`, so `dQ/dt = -r1 Q + (2 exp(-a1 tau) - 1) m(Q) Q`. The proliferating cells follow
    `dP/dt = -a1 P + (1 - exp(-a1 tau)) m(Q) Q`, the mature cells, which live `lifespan` days,
    `dL/dt = r1 Q - d1 L(t - lifespan)`, and their mean corpuscular volume
    `dMCV/dt = ((g1 U + MCV0) r1 Q - 0.85 MCV(t - lifespan) d1 L(t - lifespan) - 0.15 MCV d1 L) / L`. Before the start
    the mature cells and their volume keep their initial values.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `ErythropoiesisMercaptopurineSystem` read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.oncology import ErythropoiesisElement
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


class CellCycleChemotherapySystem(BaseSystem):
    """Tumour cells in cell-cycle compartments under chemotherapy agents.

    The agents kill, block, and recruit.

    Without a dormant phase the cells cycle through G0 and G1, S, and G2 and M, a killing agent kills a fraction `u1` of
    the dividing cells, and a blocking agent slows the transit through S by `u2`. With a dormant phase the cells rest in
    G0, cycle through G1 and S and through G2 and M, the killing agent kills `u1` of the dividing cells, and a
    recruiting agent speeds the exit from G0 by the factor `1 + v`. Each division gives `2 (1 - u1)` cells, so the
    tumour grows for `u1 < 1/2` and shrinks for `u1 > 1/2`, whatever the blocking and the recruitment. The parameters
    and initial cells are those of Tables 3.8 and 3.9 of Padmanabhan et al., in cell counts and per unit time. The book
    prints the agents as additive terms `2 l3 u1` and garbles the indices of the dormant system, so the system takes the
    form of Świerniak et al., where the agents act on the cells of the compartment they leave. Constant agents are
    given, untreated by default. The system has the cells `tumour.A1` to `tumour.A3`, or `tumour.A0` to `tumour.A2` with
    a dormant phase.

    Args:
        dormant:
            Whether the cells have a dormant phase, the system of Table 3.9, or not, the system of Table 3.8.

        u1:
            The fraction of dividing cells the killing agent kills.

        u2:
            The fraction by which the blocking agent slows the transit through S, without a dormant phase.

        v:
            The relative increase of the exit from G0 the recruiting agent causes, with a dormant phase.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import CellCycleChemotherapySystem
        >>> type(CellCycleChemotherapySystem(dormant=True).prepare().system_.elements[0]).__name__
        'DormantCellCycleElement'
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'ode', 'cell cycle', 'compartments')
    references = (BOOK, 'Świerniak, Ledzewicz, and Schättler 2003')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
        *,
        dormant: bool = False,
        u1: float = 0.0,
        u2: float = 0.0,
        v: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.dormant = dormant
        self.u1 = u1
        self.u2 = u2
        self.v = v

    def _build_system(self) -> System:
        """Return the parts of tumour cells in cell-cycle compartments under killing."""
        if self.dormant:
            dormant_interactions = [
                ReactionInteraction(
                    'recruitment',
                    target='tumour',
                    reactants={'A0': 1},
                    products={'A1': 1},
                    propensity=lambda s: s.target.l0 * (1 + s.target.v) * s.target.A0,
                ),
                ReactionInteraction(
                    'transit',
                    target='tumour',
                    reactants={'A1': 1},
                    products={'A2': 1},
                    propensity=lambda s: s.target.l1 * s.target.A1,
                ),
                ReactionInteraction(
                    'mitosis',
                    target='tumour',
                    reactants={'A2': 1},
                    propensity=lambda s: s.target.l2 * s.target.A2,
                ),
                RateInteraction(
                    'dormant_daughters',
                    target='tumour',
                    state='A0',
                    rate=lambda s: s.target.b0 * 2 * (1 - s.target.u) * s.target.l2 * s.target.A2,
                ),
                RateInteraction(
                    'cycling_daughters',
                    target='tumour',
                    state='A1',
                    rate=lambda s: s.target.b1 * 2 * (1 - s.target.u) * s.target.l2 * s.target.A2,
                ),
            ]
            return System(
                self.label,
                elements=[DormantCellCycleElement('tumour', u=self.u1, v=self.v)],
                interactions=dormant_interactions,
            )
        interactions = [
            ReactionInteraction(
                'g1_exit',
                target='tumour',
                reactants={'A1': 1},
                products={'A2': 1},
                propensity=lambda s: s.target.l1 * s.target.A1,
            ),
            ReactionInteraction(
                's_exit',
                target='tumour',
                reactants={'A2': 1},
                products={'A3': 1},
                propensity=lambda s: (1 - s.target.u2) * s.target.l2 * s.target.A2,
            ),
            ReactionInteraction(
                'division',
                target='tumour',
                reactants={'A3': 1},
                products={'A1': 2},
                propensity=lambda s: (1 - s.target.u1) * s.target.l3 * s.target.A3,
            ),
            ReactionInteraction(
                'kill',
                target='tumour',
                reactants={'A3': 1},
                propensity=lambda s: s.target.u1 * s.target.l3 * s.target.A3,
            ),
        ]
        return System(
            self.label,
            elements=[CellCycleElement('tumour', u1=self.u1, u2=self.u2)],
            interactions=interactions,
        )


class ErythropoiesisMercaptopurineSystem(BaseSystem):
    """The production of red cells under a constant level of 6-mercaptopurine.

    The production starts from health.

    The drug kills proliferating cells and lengthens their cycle, so fewer quiescent cells survive to mature, the mature
    red cells fall, and their mean volume rises to `MCV0 + g1 U`. The cells are in 10^12 per kg, the volume in fL, the
    drug in pMol per 8 x 10^8 red cells, and time in days. The parameters are those of Table 3.10 of Padmanabhan et al.,
    with its half saturation of 2.7 x 10^10 cells per kg written as `Q50 = 0.027`. The patient starts at the healthy
    steady state, `Q = Q50 (m0 / r1 - 1)^(1/n)`, `L = r1 Q / d1`, the book's ratio of proliferating to quiescent cells
    0.71 / 6.43, and `MCV = 85`. The book garbles equation 3.36 as `-r1 AQ (2 exp(a1 tau) - 1) m(AQ) AQ`. The system
    takes the standard form of Mackey 1978 that Noble et al. use, `dQ/dt = -r1 Q + (2 exp(-a1 tau) - 1) m(Q) Q`, in
    which cells leave to mature at the rate `r1` and the cells that enter the cycle return doubled once they survive the
    drug. Its healthy steady state gives 3.8 x 10^12 mature cells per kg, within the book's range. The system has the
    cells `patient.Q`, `patient.P`, `patient.L` and the mean volume `patient.MCV`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ErythropoiesisMercaptopurineSystem
        >>> ErythropoiesisMercaptopurineSystem(U=0.0).prepare().system_.elements[0].MCV
        85.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'dde', 'delay', 'haematopoiesis', 'leukaemia')
    references = (BOOK, 'Noble et al. 2010', 'Mackey 1978')
    until = 3000.0
    dt = 10.0

    def __init__(
        self,
        *,
        U: float = 500.0,
        r1: float = 0.00139,
        d1: float = 0.0071,
        m0: float = 1.01,
        Q50: float = 0.027,
        n: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.U = U
        self.r1 = r1
        self.d1 = d1
        self.m0 = m0
        self.Q50 = Q50
        self.n = n

    def _build_system(self) -> System:
        """Return the parts of the production of red cells under 6-mercaptopurine.

        The level of the drug is constant.
        """
        Q = self.Q50 * (self.m0 / self.r1 - 1) ** (1 / self.n)
        patient = ErythropoiesisElement(
            'patient',
            L=self.r1 * Q / self.d1,
            Q=Q,
            P=0.71 / 6.43 * Q,
            r1=self.r1,
            d1=self.d1,
            m0=self.m0,
            Q50=self.Q50,
            n=self.n,
            U=self.U,
        )
        interactions = [
            ReactionInteraction(
                'maturation',
                target='patient',
                reactants={'Q': 1},
                products={'L': 1},
                propensity=lambda s: s.target.r1 * s.target.Q,
            ),
            ReactionInteraction(
                'division',
                target='patient',
                reactants={'Q': 1},
                products={'Q': 2},
                propensity=lambda s: exp(
                    -s.target.a0
                    * s.target.U
                    * (s.target.g1 * s.target.U + s.target.MCV0)
                    * s.target.tau_h
                    / s.target.MCV_h,
                )
                * s.target.m0
                * s.target.Q50**s.target.n
                / (s.target.Q50**s.target.n + s.target.Q**s.target.n)
                * s.target.Q,
            ),
            ReactionInteraction(
                'damage',
                target='patient',
                reactants={'Q': 1},
                products={'P': 1},
                propensity=lambda s: (
                    1
                    - exp(
                        -s.target.a0
                        * s.target.U
                        * (s.target.g1 * s.target.U + s.target.MCV0)
                        * s.target.tau_h
                        / s.target.MCV_h,
                    )
                )
                * s.target.m0
                * s.target.Q50**s.target.n
                / (s.target.Q50**s.target.n + s.target.Q**s.target.n)
                * s.target.Q,
            ),
            ReactionInteraction(
                'damaged_loss',
                target='patient',
                reactants={'P': 1},
                propensity=lambda s: s.target.a0 * s.target.U * s.target.P,
            ),
            RateInteraction(
                'senescence',
                target='patient',
                state='L',
                rate=lambda s: -s.target.d1 * delayed(s.target.L, s.target.lifespan),
            ),
            RateInteraction(
                'new_volume',
                target='patient',
                state='MCV',
                rate=lambda s: (s.target.g1 * s.target.U + s.target.MCV0) * s.target.r1 * s.target.Q,
                divisor='L',
            ),
            RateInteraction(
                'senescent_volume',
                target='patient',
                state='MCV',
                rate=lambda s: -0.85
                * delayed(s.target.MCV, s.target.lifespan)
                * s.target.d1
                * delayed(s.target.L, s.target.lifespan),
                divisor='L',
            ),
            RateInteraction(
                'random_volume',
                target='patient',
                state='MCV',
                rate=lambda s: -0.15 * s.target.MCV * s.target.d1 * s.target.L,
                divisor='L',
            ),
        ]
        return System(self.label, elements=[patient], interactions=interactions)
