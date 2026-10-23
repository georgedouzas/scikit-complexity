"""Optimal therapy: the dosing of chemotherapy and immunotherapy that balances tumour burden against drug cost."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Law, Objective, Rate, Symbols
from ....core.runtime import register_model
from ....core.solvers import Pontryagin
from ....core.system import Observable, System
from ._chemotherapy import BOOK, TumourImmuneChemotherapy

MAX_DOSE = 1.0


def _tumour_share(s: Symbols) -> Expression:
    """Return the tumour cells as a share of the tumour and normal cells, `A / (A + N)`."""
    patient = s.patient
    share: Expression = patient.A / (patient.A + patient.N)
    return share


def _immune_ratio(s: Symbols) -> Expression:
    """Return the effector cells per tumour cell, `x / (y + 1)` in the scaled units."""
    site = s.site
    ratio: Expression = site.x / (site.y + 1)
    return ratio


class OptimalChemotherapy(TumourImmuneChemotherapy):
    """Immune, tumour, and normal cells under a chemotherapy whose infusion rate is a bounded control input.

    The cells and the drug follow the model of de Pillis and Radunskaya, `TumourImmuneChemotherapy`, with the
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
        >>> from skcomplex.models.biology.oncology import OptimalChemotherapy
        >>> OptimalChemotherapy('patient', w=0.5).w
        0.5
    """

    u: float = parameter(default=0.0, adaptive=True, bounds=(0.0, MAX_DOSE))
    w: float = parameter(default=0.1, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of the cells and of the drug, and the cost of the treatment.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [*super().laws(s), Objective(s.A + s.w * s.u**2 / 2, terminal=s.A, label='cost')]


class OptimalImmunotherapy(Element):
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
        >>> from skcomplex.models.biology.oncology import OptimalImmunotherapy
        >>> OptimalImmunotherapy('site').rho1
        0.1245
    """

    x: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    y: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    z: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    u1: float = parameter(default=0.0, adaptive=True, bounds=(0.0, MAX_DOSE))
    u2: float = parameter(default=0.0, adaptive=True, bounds=(0.0, MAX_DOSE))
    alpha_a: float = parameter(default=0.005, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.03, domain=Domain.NONNEGATIVE)
    rho1: float = parameter(default=0.1245, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.18, domain=Domain.POSITIVE)
    b: float = parameter(default=1e-4, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    rho2: float = parameter(default=0.025, domain=Domain.NONNEGATIVE)
    alpha3: float = parameter(default=0.01, domain=Domain.POSITIVE)
    du: float = parameter(default=10.0, domain=Domain.POSITIVE)
    a1: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=3.5, domain=Domain.NONNEGATIVE)
    w1: float = parameter(default=1.0, domain=Domain.POSITIVE)
    w2: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of effector cells, tumour cells, and IL-2, and the cost of the treatment.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x, y, z = s.x, s.y, s.z
        cost = y + s.w1 * s.u1**2 / 2 + s.w2 * s.u2**2 / 2
        return [
            Rate(x, s.alpha_a * y - s.d1 * x + s.a1 * s.u1 + s.rho1 * x * z / (1 + z)),
            Rate(y, s.r1 * y * (1 - s.b * y) - s.c1 * x * y / (1 + y)),
            Rate(z, s.a2 * s.u2 + s.rho2 * x * y / (s.alpha3 + y) - s.du * z),
            Objective(cost, terminal=y, label='cost'),
        ]


@register_model(
    field='biology/oncology',
    tags=('optimal control', 'chemotherapy', 'pontryagin', 'ode'),
    references=(BOOK, 'de Pillis and Radunskaya 2001'),
    defaults={'until': 30.0, 'dt': 0.25, 'solver': Pontryagin()},
    observables=lambda system: [Observable('tumour_share', _tumour_share)],
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
            rate `patient.u`, the running cost `patient.cost`, and the observable `tumour_share`.

    Examples:
        >>> from skcomplex.models.biology.oncology import optimal_chemotherapy
        >>> optimal_chemotherapy(w=1.0).components[0].w
        1.0
    """
    return System(components=[OptimalChemotherapy('patient', E=E, A=A, N=N, w=w)])


@register_model(
    field='biology/oncology',
    tags=('optimal control', 'immunotherapy', 'adoptive therapy', 'IL-2', 'pontryagin', 'ode'),
    references=(BOOK, 'Kirschner and Panetta 1998'),
    defaults={'until': 50.0, 'dt': 0.5, 'solver': Pontryagin(nodes=1001)},
    observables=lambda system: [Observable('immune_ratio', _immune_ratio)],
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
            and `site.u2`, the running cost `site.cost`, and the observable `immune_ratio`.

    Examples:
        >>> from skcomplex.models.biology.oncology import optimal_immunotherapy
        >>> optimal_immunotherapy(w1=0.5).components[0].w1
        0.5
    """
    return System(components=[OptimalImmunotherapy('site', x=x, y=y, alpha_a=alpha_a, w1=w1, w2=w2)])
