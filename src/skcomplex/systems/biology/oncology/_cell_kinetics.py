"""Oncology systems: cell-cycle compartments and red-cell production under chemotherapy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import exp

from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws.utils import delayed
from ....core.components.system import System
from ..._catalog import register_system
from ._cell_cycle import CellCycleElement
from ._dormant_cell_cycle import DormantCellCycleElement
from ._erythropoiesis import ErythropoiesisElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


@register_system(
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
    book prints the agents as additive terms `2 l3 u1` and garbles the indices of the dormant system, so the system
    takes the form of Świerniak et al., where the agents act on the cells of the compartment they leave. Constant
    agents are given, untreated by default.

    Args:
        dormant:
            Whether the cells have a dormant phase, the system of Table 3.9, or not, the system of Table 3.8.

        u1:
            The fraction of dividing cells the killing agent kills.

        u2:
            The fraction by which the blocking agent slows the transit through S, without a dormant phase.

        v:
            The relative increase of the exit from G0 the recruiting agent causes, with a dormant phase.

    Returns:
        system:
            The system, with the cells `tumour.A1` to `tumour.A3`, or `tumour.A0` to `tumour.A2` with a dormant phase.

    Examples:
        >>> from skcomplex.systems.biology.oncology import cell_cycle_chemotherapy
        >>> type(cell_cycle_chemotherapy(dormant=True).elements[0]).__name__
        'DormantCellCycleElement'
    """
    if dormant:
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
        return System(elements=[DormantCellCycleElement('tumour', u=u1, v=v)], interactions=dormant_interactions)
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
    return System(elements=[CellCycleElement('tumour', u1=u1, u2=u2)], interactions=interactions)


@register_system(
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
    0.71 / 6.43, and `MCV = 85`. The book garbles equation 3.36 as `-r1 AQ (2 exp(a1 tau) - 1) m(AQ) AQ`. The system
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
            The system, with the cells `patient.Q`, `patient.P`, `patient.L` and the mean volume `patient.MCV`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import erythropoiesis_mercaptopurine
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
    return System(elements=[patient], interactions=interactions)
