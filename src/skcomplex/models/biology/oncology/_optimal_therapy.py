"""Optimal therapy: the dosing of chemotherapy and immunotherapy that balances tumour burden against drug cost."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import RateInteraction, ReactionInteraction
from ....core.system import System
from ....engine.solvers import PontryaginSolver
from ....specs import ObservableSpec
from ..._catalog import register_model
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._chemotherapy import BOOK
from ._optimal_chemotherapy import OptimalChemotherapyElement
from ._optimal_immunotherapy import OptimalImmunotherapyElement


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
    site = OptimalImmunotherapyElement('site', x=x, y=y, alpha_a=alpha_a, w1=w1, w2=w2)
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
    return System(elements=[site], interactions=interactions)
