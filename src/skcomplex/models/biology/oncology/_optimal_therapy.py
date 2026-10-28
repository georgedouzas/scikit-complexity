"""Optimal therapy: the dosing of chemotherapy and immunotherapy that balances tumour burden against drug cost."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....engine.solvers import PontryaginSolver
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._chemotherapy import BOOK, TumourImmuneChemotherapyElement

MAX_DOSE = 1.0


def _tumour_share(s: SimpleNamespace) -> Expression:
    """Return the tumour cells as a share of the tumour and normal cells, `A / (A + N)`."""
    patient = s.patient
    share: Expression = patient.A / (patient.A + patient.N)
    return share


def _immune_ratio(s: SimpleNamespace) -> Expression:
    """Return the effector cells per tumour cell, `x / (y + 1)` in the scaled units."""
    site = s.site
    ratio: Expression = site.x / (site.y + 1)
    return ratio


def _chemotherapy_cost(s: SimpleNamespace) -> Expression:
    """Return the running cost of the chemotherapy, the tumour and the squared dose, `A + w u^2 / 2`."""
    patient = s.patient
    cost: Expression = patient.A + patient.w * patient.u**2 / 2
    return cost


def _tumour(s: SimpleNamespace) -> Expression:
    """Return the tumour cells left at the end of the chemotherapy, `A(T)`."""
    tumour: Expression = s.patient.A
    return tumour


def _immunotherapy_cost(s: SimpleNamespace) -> Expression:
    """Return the running cost of the immunotherapy, the tumour and the squared doses of both therapies."""
    site = s.site
    cost: Expression = site.y + site.w1 * site.u1**2 / 2 + site.w2 * site.u2**2 / 2
    return cost


def _tumour_site(s: SimpleNamespace) -> Expression:
    """Return the tumour cells left at the end of the immunotherapy, `y(T)`."""
    tumour: Expression = s.site.y
    return tumour


class OptimalChemotherapyElement(TumourImmuneChemotherapyElement):
    """Immune, tumour, and normal cells under a chemotherapy whose infusion rate is a bounded control input.

    The cells and the drug follow the model of de Pillis and Radunskaya, `TumourImmuneChemotherapyElement`, with the
    infusion rate `u` between 0 and its largest dose. The oncologist minimises the tumour burden and the toxicity of
    the drug, `∫ (A + w u^2 / 2) dt + A(T)`, where the terminal term weighs the tumour left at the end of the
    treatment.

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

        u:
            The infusion rate of the drug, the control input.

        w:
            The weight of the squared infusion rate in the cost.

    Examples:
        >>> from skcomplex.models.biology.oncology import OptimalChemotherapyElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the balances of the cells and of the drug.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return super()._get_laws(s)


class OptimalImmunotherapyElement(Element):
    """Effector cells, tumour cells, and IL-2 under adoptive therapy and IL-2 injection at bounded rates.

    The Kirschner-Panetta model in scaled units, with the effector and tumour cells in units of `10^5` and IL-2 in
    units of `2 10^7`, follows `x' = alpha_a y - d1 x + a1 u1 + rho1 x z / (1 + z)` for the effector cells,
    `y' = r1 y (1 - b y) - c1 x y / (1 + y)` for the tumour, and `z' = a2 u2 + rho2 x y / (alpha3 + y) - du z` for
    IL-2. The adoptive therapy `u1` and the IL-2 injection `u2` are control inputs between 0 and 1, and the oncologist
    minimises `∫ (y + w1 u1^2 / 2 + w2 u2^2 / 2) dt + y(T)`.

    Args:
        label:
            The label of the tumour site.

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
        >>> from skcomplex.models.biology.oncology import OptimalImmunotherapyElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the balances of effector cells, tumour cells, and IL-2.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x, y, z = s.x, s.y, s.z
        return [
            RateLaw(x, s.alpha_a * y - s.d1 * x + s.a1 * s.u1 + s.rho1 * x * z / (1 + z)),
            RateLaw(y, s.r1 * y * (1 - s.b * y) - s.c1 * x * y / (1 + y)),
            RateLaw(z, s.a2 * s.u2 + s.rho2 * x * y / (s.alpha3 + y) - s.du * z),
        ]


@register_model(
    field='biology/oncology',
    tags=('optimal control', 'chemotherapy', 'pontryagin', 'ode'),
    references=(BOOK, 'de Pillis and Radunskaya 2001'),
    defaults={'until': 30.0, 'dt': 0.25, 'solver': PontryaginSolver(objective=_chemotherapy_cost, terminal=_tumour)},
    observables=lambda system: [ObservableSpec('tumour_share', _tumour_share)],
)
def optimal_chemotherapy(
    E: float = 0.15,
    A: float = 0.25,
    N: float = 1.0,
    *,
    w: float = 0.1,
) -> System:
    """Return a patient under the chemotherapy that minimises the tumour burden and the toxicity of the drug.

    Run with an optimal control solver, the infusion is largest at the start, when the tumour is largest, and falls as
    the tumour shrinks, so the tumour ends far below its untreated level.

    Args:
        E:
            The initial immune cells.

        A:
            The initial tumour cells.

        N:
            The initial normal cells.

        w:
            The weight of the squared infusion rate in the cost.

    Returns:
        system:
            The model, with the cells `patient.E`, `patient.A`, and `patient.N`, the drug `patient.U`, the infusion
            rate `patient.u`, and the observable `tumour_share`.

    Examples:
        >>> from skcomplex.models.biology.oncology import optimal_chemotherapy
        >>> optimal_chemotherapy(w=1.0).elements[0].w
        1.0
    """
    return System(elements=[OptimalChemotherapyElement('patient', E=E, A=A, N=N, w=w)])


@register_model(
    field='biology/oncology',
    tags=('optimal control', 'immunotherapy', 'adoptive therapy', 'IL-2', 'pontryagin', 'ode'),
    references=(BOOK, 'Kirschner and Panetta 1998'),
    defaults={
        'until': 50.0,
        'dt': 0.5,
        'solver': PontryaginSolver(objective=_immunotherapy_cost, terminal=_tumour_site, nodes=1001),
    },
    observables=lambda system: [ObservableSpec('immune_ratio', _immune_ratio)],
)
def optimal_immunotherapy(
    x: float = 0.1,
    y: float = 1.0,
    *,
    alpha_a: float = 0.005,
    w1: float = 1.0,
    w2: float = 1.0,
) -> System:
    """Return a tumour under the adoptive therapy and IL-2 injection that minimise its burden and their cost.

    Run with an optimal control solver, the therapies are strongest while the tumour is large, the effector cells
    expand, and the tumour shrinks.

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

    Returns:
        system:
            The model, with the effector cells `site.x`, the tumour `site.y`, IL-2 `site.z`, the therapies `site.u1`
            and `site.u2`, and the observable `immune_ratio`.

    Examples:
        >>> from skcomplex.models.biology.oncology import optimal_immunotherapy
        >>> optimal_immunotherapy(w1=0.5).elements[0].w1
        0.5
    """
    return System(elements=[OptimalImmunotherapyElement('site', x=x, y=y, alpha_a=alpha_a, w1=w1, w2=w2)])
