"""Optimal therapy of chemotherapy and immunotherapy.

The dosing balances tumour burden against drug cost.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import PontryaginSolver
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._tumour_immune_chemotherapy import TumourImmuneChemotherapyElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'

MAX_DOSE = 1.0


class OptimalChemotherapyElement(TumourImmuneChemotherapyElement):
    """Immune, tumour, and normal cells under a chemotherapy.

    The infusion rate of the chemotherapy is a bounded control input.

    The cells and the drug follow the system of de Pillis and Radunskaya, `TumourImmuneChemotherapyElement`, with the
    infusion rate `u` between 0 and its largest dose. The oncologist minimises the tumour burden and the toxicity of
    the drug, `∫ (A + w u^2 / 2) dt + A(T)`, where the terminal term weighs the tumour left at the end of the
    treatment.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
            The infusion rate of the drug, the control input.

        w:
            The weight of the squared infusion rate in the cost.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OptimalChemotherapyElement
        >>> OptimalChemotherapyElement('patient', w=0.5).w
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: float = 0.15,
        A: float = 0.25,
        N: float = 1.0,
        U: float = 0.0,
        s: float = 0.33,
        rho: float = 0.01,
        alpha: float = 0.3,
        c1: float = 1.0,
        c2: float = 0.5,
        c3: float = 1.0,
        c4: float = 1.0,
        d1: float = 0.2,
        r1: float = 1.5,
        r2: float = 1.0,
        b1: float = 1.0,
        b2: float = 1.0,
        a1: float = 0.2,
        a2: float = 0.3,
        a3: float = 0.1,
        k: float = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_DOSE))] = 0.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
            E=E,
            A=A,
            N=N,
            U=U,
            s=s,
            rho=rho,
            alpha=alpha,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            d1=d1,
            r1=r1,
            r2=r2,
            b1=b1,
            b2=b2,
            a1=a1,
            a2=a2,
            a3=a3,
            k=k,
            u=u,
        )
        self.w = w


class OptimalImmunotherapyElement(Element):
    """Effector and tumour cells and IL-2 under adoptive therapy and IL-2 injection.

    The rates of both therapies are bounded.

    The Kirschner-Panetta system in scaled units, with the effector and tumour cells in units of `10^5` and IL-2 in
    units of `2 10^7`, follows `x' = alpha_a y - d1 x + a1 u1 + rho1 x z / (1 + z)` for the effector cells,
    `y' = r1 y (1 - b y) - c1 x y / (1 + y)` for the tumour, and `z' = a2 u2 + rho2 x y / (alpha3 + y) - du z` for
    IL-2. The adoptive therapy `u1` and the IL-2 injection `u2` are control inputs between 0 and 1, and the oncologist
    minimises `∫ (y + w1 u1^2 / 2 + w2 u2^2 / 2) dt + y(T)`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `OptimalImmunotherapySystem` read its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial effector cells.

        y:
            The initial tumour cells.

        z:
            The initial IL-2 concentration.

        u1:
            The adoptive therapy rate, a control input.

        u2:
            The IL-2 injection rate, a control input.

        alpha_a:
            The antigenicity of the tumour.

        d1:
            The death rate of effector cells.

        rho1:
            The proliferation rate of effector cells stimulated by IL-2.

        r1:
            The growth rate of the tumour.

        b:
            The inverse carrying capacity of the tumour.

        c1:
            The kill rate of tumour cells by effector cells.

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

        w1:
            The weight of the squared adoptive therapy rate in the cost.

        w2:
            The weight of the squared IL-2 injection rate in the cost.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OptimalImmunotherapyElement
        >>> OptimalImmunotherapyElement('site').rho1
        0.1245
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u1: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_DOSE))] = 0.0,
        u2: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_DOSE))] = 0.0,
        alpha_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        alpha3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.5,
        w1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        w2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.u1 = u1
        self.u2 = u2
        self.alpha_a = alpha_a
        self.d1 = d1
        self.rho1 = rho1
        self.r1 = r1
        self.b = b
        self.c1 = c1
        self.rho2 = rho2
        self.alpha3 = alpha3
        self.du = du
        self.a1 = a1
        self.a2 = a2
        self.w1 = w1
        self.w2 = w2


def _tumour_share(s: SimpleNamespace) -> Expression:
    """Return the tumour cells as a share of the tumour and normal cells.

    The share is `A / (A + N)`.
    """
    patient = s.patient
    share: Expression = patient.A / (patient.A + patient.N)
    return share


def _immune_ratio(s: SimpleNamespace) -> Expression:
    """Return the effector cells per tumour cell, `x / (y + 1)` in the scaled units."""
    site = s.site
    ratio: Expression = site.x / (site.y + 1)
    return ratio


def _chemotherapy_cost(s: SimpleNamespace) -> Expression:
    """Return the running cost of the chemotherapy, `A + w u^2 / 2`.

    It sums the tumour and the squared dose.
    """
    patient = s.patient
    cost: Expression = patient.A + patient.w * patient.u**2 / 2
    return cost


def _tumour(s: SimpleNamespace) -> Expression:
    """Return the tumour cells left at the end of the chemotherapy, `A(T)`."""
    tumour: Expression = s.patient.A
    return tumour


def _immunotherapy_cost(s: SimpleNamespace) -> Expression:
    """Return the running cost of the immunotherapy.

    It sums the tumour and the squared doses of both therapies.
    """
    site = s.site
    cost: Expression = site.y + site.w1 * site.u1**2 / 2 + site.w2 * site.u2**2 / 2
    return cost


def _tumour_site(s: SimpleNamespace) -> Expression:
    """Return the tumour cells left at the end of the immunotherapy, `y(T)`."""
    tumour: Expression = s.site.y
    return tumour


class OptimalChemotherapySystem(BaseSystem):
    """A patient under the chemotherapy that minimises tumour and toxicity.

    The chemotherapy minimises the tumour burden and the toxicity of the drug.

    Run with an optimal control solver, the infusion is largest at the start, when the tumour is largest, and falls as
    the tumour shrinks, so the tumour ends far below its untreated level. The system has the cells `patient.E`,
    `patient.A`, and `patient.N`, the drug `patient.U`, the infusion rate `patient.u`, and the observable
    `tumour_share`.

    Args:
        E:
            The initial immune cells.

        A:
            The initial tumour cells.

        N:
            The initial normal cells.

        w:
            The weight of the squared infusion rate in the cost.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OptimalChemotherapySystem
        >>> OptimalChemotherapySystem(w=1.0).prepare().system_.elements[0].w
        1.0
    """

    field = 'biology/oncology'
    topics = ('optimal control', 'chemotherapy', 'pontryagin', 'ode')
    references = (BOOK, 'de Pillis and Radunskaya 2001')
    until = 30.0
    dt = 0.25
    solver = PontryaginSolver(objective=_chemotherapy_cost, terminal=_tumour)
    observables = (ObservableSpec('tumour_share', _tumour_share),)

    def __init__(
        self,
        E: float = 0.15,
        A: float = 0.25,
        N: float = 1.0,
        *,
        w: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.E = E
        self.A = A
        self.N = N
        self.w = w

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a patient under the chemotherapy."""
        return CompositeSystem(
            self.label,
            elements=[OptimalChemotherapyElement('patient', E=self.E, A=self.A, N=self.N, w=self.w)],
        )


class OptimalImmunotherapySystem(BaseSystem):
    """A tumour under the adoptive therapy and IL-2 injection that minimise it.

    The therapies minimise its burden and their cost.

    Run with an optimal control solver, the therapies are strongest while the tumour is large, the effector cells
    expand, and the tumour shrinks. The system has the effector cells `site.x`, the tumour `site.y`, IL-2 `site.z`, the
    therapies `site.u1` and `site.u2`, and the observable `immune_ratio`.

    Args:
        x:
            The initial effector cells, in units of `10^5`.

        y:
            The initial tumour cells, in units of `10^5`.

        alpha_a:
            The antigenicity of the tumour.

        w1:
            The weight of the squared adoptive therapy rate in the cost.

        w2:
            The weight of the squared IL-2 injection rate in the cost.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OptimalImmunotherapySystem
        >>> OptimalImmunotherapySystem(w1=0.5).prepare().system_.elements[0].w1
        0.5
    """

    field = 'biology/oncology'
    topics = ('optimal control', 'immunotherapy', 'adoptive therapy', 'IL-2', 'pontryagin', 'ode')
    references = (BOOK, 'Kirschner and Panetta 1998')
    until = 50.0
    dt = 0.5
    solver = PontryaginSolver(objective=_immunotherapy_cost, terminal=_tumour_site, nodes=1001)
    observables = (ObservableSpec('immune_ratio', _immune_ratio),)

    def __init__(
        self,
        x: float = 0.1,
        y: float = 1.0,
        *,
        alpha_a: float = 0.005,
        w1: float = 1.0,
        w2: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.alpha_a = alpha_a
        self.w1 = w1
        self.w2 = w2

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a tumour under the adoptive therapy and IL-2 injection."""
        site = OptimalImmunotherapyElement('site', x=self.x, y=self.y, alpha_a=self.alpha_a, w1=self.w1, w2=self.w2)
        interactions = [
            RateInteraction('stimulation', target='site', state='x', rate=lambda s: s.target.alpha_a * s.target.y),
            ReactionInteraction(
                'effector_death',
                target='site',
                reactants={'x': 1},
                propensity=lambda s: s.target.d1 * s.target.x,
            ),
            DosingInteraction('adoptive_therapy', target='site', state='x', dose=lambda s: s.target.a1 * s.target.u1),
            MichaelisMentenInteraction(
                'effector_proliferation',
                target='site',
                products={'x': 1},
                substrate='z',
                vmax=lambda s: s.target.rho1,
                km=1.0,
                modifiers={'x': 1},
            ),
            RateInteraction(
                'tumour_growth',
                target='site',
                state='y',
                rate=lambda s: s.target.r1 * s.target.y * (1 - s.target.b * s.target.y),
            ),
            MichaelisMentenInteraction(
                'kill',
                target='site',
                reactants={'y': 1},
                substrate='y',
                vmax=lambda s: s.target.c1,
                km=1.0,
                modifiers={'x': 1},
            ),
            DosingInteraction('il2_injection', target='site', state='z', dose=lambda s: s.target.a2 * s.target.u2),
            MichaelisMentenInteraction(
                'il2_production',
                target='site',
                products={'z': 1},
                substrate='y',
                vmax=lambda s: s.target.rho2,
                km=lambda s: s.target.alpha3,
                modifiers={'x': 1},
            ),
            ReactionInteraction(
                'il2_loss',
                target='site',
                reactants={'z': 1},
                propensity=lambda s: s.target.du * s.target.z,
            ),
        ]
        return CompositeSystem(self.label, elements=[site], interactions=interactions)
