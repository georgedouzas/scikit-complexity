"""Tumour, immune, and normal cells under chemotherapy.

The systems cover resistance, delays, and metastasis.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, RateInteraction
from ....core.components.laws.utils import clip, delayed
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ...general.reactions import ReactionInteraction
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._tumour_immune_chemotherapy import TumourImmuneChemotherapyElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'
DRUGS = (1, 2, 3)
RESISTANCE_CLASSES = {
    (): 'S',
    (1,): 'R1',
    (2,): 'R2',
    (3,): 'R3',
    (1, 2): 'R12',
    (1, 3): 'R13',
    (2, 3): 'R23',
    (1, 2, 3): 'R123',
}


class MetastasisChemotherapyElement(Element):
    """Normal and tumour cells at a primary and a secondary site.

    Each site has its drug, and the migration between them is delayed.

    At each site the normal and tumour cells grow logistically, compete, and a drug kills them with a saturating
    response, `a N U / (bc + N)`. The drug at a site follows `dU/dt = u - (k + g1 N / (bc + N) + g2 A / (bc + A)) U`.
    Tumour cells leave the primary site at the rate `delta`, and a fraction `epsilon` of them reaches the secondary site
    after the delay `tau`, `dAs/dt = ... + epsilon delta Ap(t - tau)`. Before the start the primary tumour keeps its
    initial value.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `DelayedMetastasisSystem` read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.oncology import MetastasisChemotherapyElement
        >>> MetastasisChemotherapyElement('patient').epsilon
        0.0001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Np: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1460.0,
        Ap: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 165.0,
        Up: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Ns: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1750.0,
        As: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Us: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 12.0,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00068,
        b2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00047,
        b3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00057,
        b4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00043,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0075,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0008,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0006,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        g1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0024,
        g2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.24,
        g3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0018,
        g4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        bc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        up: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        us: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Np = Np
        self.Ap = Ap
        self.Up = Up
        self.Ns = Ns
        self.As = As
        self.Us = Us
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4
        self.bc = bc
        self.k = k
        self.delta = delta
        self.epsilon = epsilon
        self.tau = tau
        self.up = up
        self.us = us


class MultidrugResistanceElement(Element):
    """Tumour cells split by their resistance to three drugs.

    Each drug is given by a constant infusion with its toxicity.

    The sensitive cells `S`, the cells resistant to one drug `R1`, `R2`, `R3`, to two drugs `R12`, `R13`, `R23`, and
    to all three `R123` grow at the same rate `r`. A fraction `m_i` of the offspring of a population gains resistance
    to drug `i`, so `dS/dt = r (1 - m1 - m2 - m3) S - K S`, `dRi/dt = r ((1 - mj - mk) Ri + mi S) - K Ri`, and so on,
    where `K` sums the kill `a_i (U_i - Uth) F(U_i - Uth)` of every drug the population is sensitive to. The drugs
    follow `dU_i/dt = u_i - k_i U_i` and their toxicities `dTX_i/dt = U_i - eta_i TX_i`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `MultidrugResistanceSystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.oncology import MultidrugResistanceElement
        >>> MultidrugResistanceElement('tumour').Uth
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 460500000000.0,
        R1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R12: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R13: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R23: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R123: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        TX1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        TX2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        TX3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0099,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.008,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        m3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.014,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0084,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0074,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0092,
        Uth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        k1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.32,
        k2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.27,
        k3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        eta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        eta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        eta3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.45,
        u1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R12 = R12
        self.R13 = R13
        self.R23 = R23
        self.R123 = R123
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.TX1 = TX1
        self.TX2 = TX2
        self.TX3 = TX3
        self.r = r
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.Uth = Uth
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3


class RestingHuntingChemotherapyElement(Element):
    """Hunting immune cells recruited from resting cells after a delay, and a tumour.

    A drug kills all three.

    The hunting cells follow `dE/dt = m1 E Q(t - tau) - d1 E - c1 A E - a E / (bd + E) U`, the resting cells
    `dQ/dt = r1 Q (1 - b1 Q) - m1 E Q(t - tau) - a Q / (bd + Q) U`, the tumour
    `dA/dt = r2 A (1 - b2 A) - c2 A E - a A / (bd + A) U`, and the drug
    `dU/dt = u - (k + g (E / (bd + E) + Q / (bd + Q) + A / (bd + A))) U`, which binds to the cells it kills. Before
    the start the resting cells keep their initial value.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `DelayedRestingHuntingChemotherapySystem` read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.oncology import RestingHuntingChemotherapyElement
        >>> RestingHuntingChemotherapyElement('patient').tau
        45.6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        Q: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.48,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.093,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0412,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005133,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.6515,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0245,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        b2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        bd: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0001,
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 45.6,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.Q = Q
        self.A = A
        self.U = U
        self.m1 = m1
        self.d1 = d1
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.b1 = b1
        self.r2 = r2
        self.b2 = b2
        self.a = a
        self.bd = bd
        self.g = g
        self.k = k
        self.tau = tau
        self.u = u


class DePillisRadunskayaSystem(BaseSystem):
    """The de Pillis-Radunskaya system of immune, tumour, and normal cells.

    The cells are under a constant drug infusion.

    The cells are normalised so that the carrying capacity of normal cells is 1, and time is in days. Untreated, a
    tumour that starts at 0.25 escapes the immune cells and settles at the high equilibrium where it outnumbers the
    normal cells. A constant infusion `u = 1` clears it, while the normal cells stay above 0.7. The parameters are those
    of Table 3.3 of Padmanabhan et al. The book's initial tumour of `1e-5` lies in the basin of the tumour-free
    equilibrium, which is stable since `r1 < c2 s / d1 + c3`, so the defaults are the initial conditions of de Pillis
    and Radunskaya 2003, `E = 0.15`, `A = 0.25`, `N = 1`. The book prints no infusion schedule, so the infusion is a
    constant rate `u` per day, untreated by default. The system has the cells `patient.E`, `patient.A`, `patient.N` and
    the drug `patient.U`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DePillisRadunskayaSystem
        >>> DePillisRadunskayaSystem(u=1.0).prepare().system_.elements[0].u
        1.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'ode', 'tumour-immune', 'competition')
    references = (BOOK, 'de Pillis and Radunskaya 2003')
    until = 150.0
    dt = 0.5

    def __init__(
        self,
        E: float = 0.15,
        A: float = 0.25,
        N: float = 1.0,
        *,
        u: float = 0.0,
        s: float = 0.33,
        r1: float = 1.5,
        a2: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.A = A
        self.N = N
        self.u = u
        self.s = s
        self.r1 = r1
        self.a2 = a2

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the de Pillis-Radunskaya system of immune."""
        return CompositeSystem(
            self.label,
            elements=[
                TumourImmuneChemotherapyElement(
                    'patient',
                    E=self.E,
                    A=self.A,
                    N=self.N,
                    u=self.u,
                    s=self.s,
                    r1=self.r1,
                    a2=self.a2,
                ),
            ],
        )


class MultidrugResistanceSystem(BaseSystem):
    """A tumour of eight populations resistant to subsets of three drugs.

    The drugs are given by constant infusions.

    Each drug kills the populations sensitive to it once its concentration exceeds a threshold, so the drugs clear the
    sensitive cells while the cells resistant to all three keep growing. Time is in days, the cells are counted, and the
    drugs are in dosage units D. The parameters are those of Table 3.4 of Padmanabhan et al. The book prints the
    triple-resistant population without its own growth, `dR123/dt = r (m1 R23 + m2 R13 + m3 R12)`, against its
    assumption that all populations grow at the same rate, so the system adds the growth `r R123`. The book prints no
    infusion schedule, so the drugs are constant infusions whose concentrations settle within the book's range of 0 to
    50. The system has the populations `tumour.S`, `tumour.R1` to `tumour.R123`, the drugs `tumour.U1` to `tumour.U3`,
    and their toxicities `tumour.TX1` to `tumour.TX3`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import MultidrugResistanceSystem
        >>> MultidrugResistanceSystem(u2=0.0).prepare().system_.elements[0].u2
        0.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'ode', 'drug resistance', 'toxicity')
    references = (BOOK, 'Tse et al. 2007')
    until = 300.0
    dt = 1.0

    def __init__(
        self,
        S: float = 4.605e11,
        *,
        r: float = 0.0099,
        u1: float = 10.0,
        u2: float = 10.0,
        u3: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.S = S
        self.r = r
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a tumour of eight populations resistant to three drugs.

        The populations are resistant to subsets of the drugs.
        """
        tumour = MultidrugResistanceElement('tumour', S=self.S, r=self.r, u1=self.u1, u2=self.u2, u3=self.u3)

        def growth(name: str) -> RateInteraction:
            """Return the division of the cells of a resistance class."""
            return RateInteraction(
                f'growth_{name}',
                target='tumour',
                state=name,
                rate=lambda s: s.target.r * getattr(s.target, name),
            )

        def mutation(name: str, drug: int, mutant: str) -> ReactionInteraction:
            """Return the offspring of a resistance class that gain one more resistance.

            The resistance gained is to one more drug.
            """
            return ReactionInteraction(
                f'mutation_{name}_{drug}',
                target='tumour',
                reactants={name: 1},
                products={mutant: 1},
                propensity=lambda s: s.target.r * getattr(s.target, f'm{drug}') * getattr(s.target, name),
            )

        def kill(name: str, drug: int) -> ReactionInteraction:
            """Return the kill of a resistance class by a drug it is sensitive to.

            The drug kills above the threshold.
            """
            return ReactionInteraction(
                f'kill_{name}_{drug}',
                target='tumour',
                reactants={name: 1},
                modifiers={f'U{drug}': 1},
                propensity=lambda s: (
                    getattr(s.target, f'a{drug}')
                    * clip(getattr(s.target, f'U{drug}') - s.target.Uth, 0)
                    * getattr(s.target, name)
                ),
            )

        def pharmacokinetics(drug: int) -> list[BaseInteraction]:
            """Return the infusion and clearance of a drug, and its toxicity.

            The toxicity builds up and recovers.
            """
            return [
                DosingInteraction(
                    f'infusion_{drug}',
                    target='tumour',
                    state=f'U{drug}',
                    dose=lambda s: getattr(s.target, f'u{drug}'),
                ),
                ReactionInteraction(
                    f'clearance_{drug}',
                    target='tumour',
                    reactants={f'U{drug}': 1},
                    propensity=lambda s: getattr(s.target, f'k{drug}') * getattr(s.target, f'U{drug}'),
                ),
                RateInteraction(
                    f'toxicity_{drug}',
                    target='tumour',
                    state=f'TX{drug}',
                    rate=lambda s: getattr(s.target, f'U{drug}'),
                ),
                ReactionInteraction(
                    f'recovery_{drug}',
                    target='tumour',
                    reactants={f'TX{drug}': 1},
                    propensity=lambda s: getattr(s.target, f'eta{drug}') * getattr(s.target, f'TX{drug}'),
                ),
            ]

        growths = [growth(name) for name in RESISTANCE_CLASSES.values()]
        sensitivities = [
            (resistances, name, drug)
            for resistances, name in RESISTANCE_CLASSES.items()
            for drug in DRUGS
            if drug not in resistances
        ]
        mutations = [
            mutation(name, drug, RESISTANCE_CLASSES[tuple(sorted((*resistances, drug)))])
            for resistances, name, drug in sensitivities
        ]
        kills = [kill(name, drug) for _, name, drug in sensitivities]
        drugs = [interaction for drug in DRUGS for interaction in pharmacokinetics(drug)]
        return CompositeSystem(self.label, elements=[tumour], interactions=[*growths, *mutations, *kills, *drugs])


class DrugResistanceSwitchingSystem(DynamicalSystem):
    """Sensitive cells and cells resistant by mutation or by a phenotype switch.

    The switch is reversible, and the cells are under a dose.

    The drug kills the sensitive cells, so the resistant cells, which grow slower, take the tumour over. Resistance is
    acquired spontaneously at the rates `m1` and `m2` and induced by the drug at the rates `alpha_m` and `alpha_p` per
    unit of dose, and switched cells return to sensitivity at the rate `m5`. Time and the cells are dimensionless, the
    carrying capacity being 1. The book gives values only for the two-population system of Greene, Sanchez-Tapia, and
    Sontag, its Table 3.5, which is this system without switched cells. The defaults are that system, with the induced
    rate `alpha_m = 0.01` of the table and an effective dose of 1, so switching is off until `m2`, `alpha_p`, or `m5` is
    set. The system has the cells `tumour.S`, `tumour.Rm`, and `tumour.Rp`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DrugResistanceSwitchingSystem
        >>> DrugResistanceSwitchingSystem(alpha_m=0.0).prepare().system_.elements[0].get_value('alpha_m')
        0.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'ode', 'drug resistance', 'phenotype switching')
    references = (BOOK, 'Greene, Sanchez-Tapia, and Sontag 2018', 'Greene, Gevertz, and Sontag 2019')
    until = 400.0
    dt = 1.0

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.S = S
        self.u = u
        self.m1 = m1
        self.m2 = m2
        self.alpha_m = alpha_m
        self.alpha_p = alpha_p
        self.m5 = m5
        self.r2 = r2
        self.r3 = r3

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'tumour'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'S': DeclaredVariableSpec(self.S, StateSpec(domain=nonnegative)),
            'Rm': DeclaredVariableSpec(0.0, StateSpec(domain=nonnegative)),
            'Rp': DeclaredVariableSpec(0.0, StateSpec(domain=nonnegative)),
            'r1': DeclaredVariableSpec(1.0, ParameterSpec(domain=nonnegative)),
            'r2': DeclaredVariableSpec(self.r2, ParameterSpec(domain=nonnegative)),
            'r3': DeclaredVariableSpec(self.r3, ParameterSpec(domain=nonnegative)),
            'b': DeclaredVariableSpec(1.0, ParameterSpec(domain=nonnegative)),
            'm1': DeclaredVariableSpec(self.m1, ParameterSpec(domain=nonnegative)),
            'm2': DeclaredVariableSpec(self.m2, ParameterSpec(domain=nonnegative)),
            'alpha_m': DeclaredVariableSpec(self.alpha_m, ParameterSpec(domain=nonnegative)),
            'alpha_p': DeclaredVariableSpec(self.alpha_p, ParameterSpec(domain=nonnegative)),
            'm5': DeclaredVariableSpec(self.m5, ParameterSpec(domain=nonnegative)),
            'a': DeclaredVariableSpec(1.0, ParameterSpec(domain=nonnegative)),
            'am': DeclaredVariableSpec(0.0, ParameterSpec(domain=nonnegative)),
            'ap': DeclaredVariableSpec(0.0, ParameterSpec(domain=nonnegative)),
            'u': DeclaredVariableSpec(self.u, ParameterSpec(domain=nonnegative)),
        }

    def _get_equations(self) -> Equations:
        """Return the growth, transitions, and kill of the sensitive, mutated, and switched cells."""
        return {'S': self._sensitive_rate, 'Rm': self._mutated_rate, 'Rp': self._switched_rate}

    @staticmethod
    def _room(tumour: SimpleNamespace) -> Expression:
        """Return the room left for growth, `1 - b V`."""
        return 1 - tumour.b * (tumour.S + tumour.Rm + tumour.Rp)

    @staticmethod
    def _sensitive_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of the sensitive cells."""
        tumour = s.target
        room = DrugResistanceSwitchingSystem._room(tumour)
        return (
            tumour.r1 * room * tumour.S
            - (tumour.m1 + tumour.m2) * tumour.S
            - (tumour.alpha_m + tumour.alpha_p) * tumour.u * tumour.S
            - tumour.a * tumour.u * tumour.S
            + tumour.m5 * tumour.Rp
        )

    @staticmethod
    def _mutated_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of the cells resistant by mutation."""
        tumour = s.target
        room = DrugResistanceSwitchingSystem._room(tumour)
        return (
            tumour.r2 * room * tumour.Rm
            + tumour.m1 * tumour.S
            + tumour.alpha_m * tumour.u * tumour.S
            - tumour.am * tumour.u * tumour.Rm
        )

    @staticmethod
    def _switched_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of the cells resistant by phenotype switching."""
        tumour = s.target
        room = DrugResistanceSwitchingSystem._room(tumour)
        return (
            tumour.r3 * room * tumour.Rp
            + tumour.m2 * tumour.S
            + tumour.alpha_p * tumour.u * tumour.S
            - tumour.m5 * tumour.Rp
            - tumour.ap * tumour.u * tumour.Rp
        )


class DelayedRestingHuntingChemotherapySystem(BaseSystem):
    """The tumour-immune system of resting and hunting cells with a conversion delay.

    The system is under a drug infusion.

    Resting immune cells become hunting cells after the delay `tau`, and hunting cells kill the tumour. Time and the
    cells are dimensionless. The parameters and initial conditions are those of Table 3.6 of Padmanabhan et al., and the
    infusion is a constant rate `u`, untreated by default, within the book's range of 0 to 0.03. The system has the
    cells `patient.E`, `patient.Q`, `patient.A` and the drug `patient.U`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DelayedRestingHuntingChemotherapySystem
        >>> DelayedRestingHuntingChemotherapySystem(tau=10.0).prepare().system_.elements[0].tau
        10.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'dde', 'delay', 'tumour-immune')
    references = (BOOK, 'Banerjee and Sarkar 2008', 'Borges et al. 2014')
    until = 1500.0
    dt = 5.0

    def __init__(
        self,
        E: float = 0.01,
        Q: float = 0.48,
        A: float = 0.18,
        *,
        u: float = 0.0,
        tau: float = 45.6,
        a: float = 0.001,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.Q = Q
        self.A = A
        self.u = u
        self.tau = tau
        self.a = a

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the tumour-immune system of resting and hunting cells.

        The conversion of the cells is delayed.
        """
        patient = RestingHuntingChemotherapyElement(
            'patient',
            E=self.E,
            Q=self.Q,
            A=self.A,
            u=self.u,
            tau=self.tau,
            a=self.a,
        )
        drug_effects = [
            interaction
            for name in ('E', 'Q', 'A')
            for interaction in (
                MichaelisMentenInteraction(
                    f'drug_kill_{name}',
                    target='patient',
                    reactants={name: 1},
                    substrate=name,
                    vmax=lambda s: s.target.a,
                    km=lambda s: s.target.bd,
                    modifiers={'U': 1},
                ),
                MichaelisMentenInteraction(
                    f'drug_uptake_{name}',
                    target='patient',
                    reactants={'U': 1},
                    substrate=name,
                    vmax=lambda s: s.target.g,
                    km=lambda s: s.target.bd,
                    modifiers={'U': 1},
                ),
            )
        ]
        interactions = [
            ReactionInteraction(
                'recruitment',
                target='patient',
                reactants={'Q': 1},
                products={'E': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.m1 * s.target.E * delayed(s.target.Q, s.target.tau),
            ),
            ReactionInteraction(
                'effector_death',
                target='patient',
                reactants={'E': 1},
                propensity=lambda s: s.target.d1 * s.target.E,
            ),
            ReactionInteraction(
                'effector_inactivation',
                target='patient',
                reactants={'E': 1},
                modifiers={'A': 1},
                propensity=lambda s: s.target.c1 * s.target.A * s.target.E,
            ),
            RateInteraction(
                'resting_growth',
                target='patient',
                state='Q',
                rate=lambda s: s.target.r1 * s.target.Q * (1 - s.target.b1 * s.target.Q),
            ),
            RateInteraction(
                'tumour_growth',
                target='patient',
                state='A',
                rate=lambda s: s.target.r2 * s.target.A * (1 - s.target.b2 * s.target.A),
            ),
            ReactionInteraction(
                'tumour_kill',
                target='patient',
                reactants={'A': 1},
                modifiers={'E': 1},
                propensity=lambda s: s.target.c2 * s.target.A * s.target.E,
            ),
            DosingInteraction('infusion', target='patient', state='U', dose=lambda s: s.target.u),
            ReactionInteraction(
                'clearance',
                target='patient',
                reactants={'U': 1},
                propensity=lambda s: s.target.k * s.target.U,
            ),
            *drug_effects,
        ]
        return CompositeSystem(self.label, elements=[patient], interactions=interactions)


class DelayedMetastasisSystem(BaseSystem):
    """Normal and tumour cells at a primary and a secondary site.

    The migration of tumour cells is delayed.

    Untreated, the faster tumour cells outcompete the normal cells at the primary site, and the cells that migrate seed
    a tumour that does the same at the secondary site. The parameters are those of Table 3.7 of Padmanabhan et al., with
    time in days, and the initial cells are at the upper ends of the book's ranges with the secondary site free of
    tumour. The book prints the kill at the secondary site with the infusion `us` in place of the concentration `Us`, so
    the system uses `Us` as at the primary site. The infusions are constant rates, untreated by default, within the
    book's range of 0 to 650. The system has the cells `patient.Np`, `patient.Ap`, `patient.Ns`, `patient.As` and the
    drugs `patient.Up` and `patient.Us`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import DelayedMetastasisSystem
        >>> DelayedMetastasisSystem(up=650.0).prepare().system_.elements[0].up
        650.0
    """

    field = 'biology/oncology'
    topics = ('chemotherapy', 'dde', 'delay', 'metastasis')
    references = (BOOK, 'Pinho, Freedman, and Nani 2002')
    until = 10.0
    dt = 0.05

    def __init__(
        self,
        Np: float = 1460.0,
        Ap: float = 165.0,
        Ns: float = 1750.0,
        As: float = 0.0,
        *,
        up: float = 0.0,
        us: float = 0.0,
        tau: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.Np = Np
        self.Ap = Ap
        self.Ns = Ns
        self.As = As
        self.up = up
        self.us = us
        self.tau = tau

    def _build_system(self) -> CompositeSystem:
        """Return the parts of normal and tumour cells at a primary and a secondary site."""
        patient = MetastasisChemotherapyElement(
            'patient',
            Np=self.Np,
            Ap=self.Ap,
            Ns=self.Ns,
            As=self.As,
            up=self.up,
            us=self.us,
            tau=self.tau,
        )

        def population(cells: str, other: str, drug: str, index: int, uptake: str) -> list[BaseInteraction]:
            """Return the growth, competition, and drug kill and uptake of the cells of a site."""
            return [
                RateInteraction(
                    f'growth_{cells}',
                    target='patient',
                    state=cells,
                    rate=lambda s: (
                        getattr(s.target, f'r{index}')
                        * getattr(s.target, cells)
                        * (1 - getattr(s.target, f'b{index}') * getattr(s.target, cells))
                    ),
                ),
                ReactionInteraction(
                    f'competition_{cells}',
                    target='patient',
                    reactants={cells: 1},
                    modifiers={other: 1},
                    propensity=lambda s: (
                        getattr(s.target, f'c{index}') * getattr(s.target, cells) * getattr(s.target, other)
                    ),
                ),
                MichaelisMentenInteraction(
                    f'drug_kill_{cells}',
                    target='patient',
                    reactants={cells: 1},
                    substrate=cells,
                    vmax=lambda s: getattr(s.target, f'a{index}'),
                    km=lambda s: s.target.bc,
                    modifiers={drug: 1},
                ),
                MichaelisMentenInteraction(
                    f'drug_uptake_{cells}',
                    target='patient',
                    reactants={drug: 1},
                    substrate=cells,
                    vmax=lambda s: getattr(s.target, uptake),
                    km=lambda s: s.target.bc,
                    modifiers={drug: 1},
                ),
            ]

        def drug(name: str, dose: str) -> list[BaseInteraction]:
            """Return the infusion and the clearance of the drug at a site."""
            return [
                DosingInteraction(
                    f'infusion_{name}',
                    target='patient',
                    state=name,
                    dose=lambda s: getattr(s.target, dose),
                ),
                ReactionInteraction(
                    f'clearance_{name}',
                    target='patient',
                    reactants={name: 1},
                    propensity=lambda s: s.target.k * getattr(s.target, name),
                ),
            ]

        interactions = [
            *population('Np', 'Ap', 'Up', 1, 'g1'),
            *population('Ap', 'Np', 'Up', 2, 'g2'),
            *population('Ns', 'As', 'Us', 3, 'g3'),
            *population('As', 'Ns', 'Us', 4, 'g4'),
            *drug('Up', 'up'),
            *drug('Us', 'us'),
        ]
        interactions += [
            ReactionInteraction(
                'emigration',
                target='patient',
                reactants={'Ap': 1},
                propensity=lambda s: s.target.delta * s.target.Ap,
            ),
            RateInteraction(
                'arrival',
                target='patient',
                state='As',
                rate=lambda s: s.target.epsilon * s.target.delta * delayed(s.target.Ap, s.target.tau),
            ),
        ]
        return CompositeSystem(self.label, elements=[patient], interactions=interactions)
