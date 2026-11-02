"""Oncology models: tumours whose vasculature an angiogenesis inhibitor prunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import log

from ....core.interactions import RateInteraction, ReactionInteraction
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model
from ..common.pharmacokinetics import DosingInteraction
from ._angiogenic_tumour import AngiogenicTumourElement
from ._vascularized_spheroid import INHIBITOR_CLEARANCE, VascularizedSpheroidElement

BOOK_REFERENCE = 'Padmanabhan, Meskin, and Al Moustafa 2021'


@register_model(
    field='biology/oncology',
    tags=('anti-angiogenic therapy', 'ode', 'angiogenesis', 'Gompertz', 'dosing'),
    references=(BOOK_REFERENCE, 'Hahnfeldt et al. 1999'),
    defaults={'until': 30.0, 'dt': 0.1},
)
def hahnfeldt(
    A: float = 17347.0,
    Vb: float = 17347.0,
    *,
    lambda1: float = 0.192,
    lambda2: float = 0.0,
    xi_s: float = 5.85,
    xi_i: float = 0.00873,
    a: float = 0.66,
    du: float = 1.3,
    dose: float = 20.0,
    period: float = 1.0,
    width: float = 1.0,
    start: float = 0.0,
) -> System:
    """Return the Hahnfeldt model, a tumour whose carrying capacity is the vasculature it stimulates and inhibits.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)` and the vasculature
    `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb A^(2/3) - a Vb U`. The inhibitor follows `dU/dt = -du U + u(t)`, the
    differential form of the book's convolution. Untreated, tumour and vasculature tend to the volume
    `(xi_s / xi_i)^(3/2)`, about 17347 mm³, which is the initial state. The inhibitor parameters are those of endostatin
    in Lewis lung carcinoma, and the default schedule is the continuous infusion of 20 mg/kg/day of Hahnfeldt et al.,
    under which the vasculature and then the tumour regress. Volumes are in mm³, time in days, and the inhibitor in
    mg/kg.

    Args:
        A:
            The initial tumour volume.

        Vb:
            The initial vascular volume.

        lambda1:
            The rate at which the tumour approaches its capacity.

        lambda2:
            The spontaneous loss rate of the vasculature.

        xi_s:
            The angiogenic stimulation by the tumour.

        xi_i:
            The angiogenic inhibition by the tumour.

        a:
            The vascular loss per unit of inhibitor.

        du:
            The clearance rate of the inhibitor.

        dose:
            The infusion rate during a dose, 0 for no treatment.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose, equal to `period` for a continuous infusion.

        start:
            The time of the first dose.

    Returns:
        system:
            The model, with the volumes `tumour.A` and `tumour.Vb` and the inhibitor `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import hahnfeldt
        >>> hahnfeldt(dose=0.0).elements[0].dose
        0.0
    """
    tumour = AngiogenicTumourElement(
        'tumour',
        A=A,
        Vb=Vb,
        lambda1=lambda1,
        lambda2=lambda2,
        xi_s=xi_s,
        xi_i=xi_i,
        a=a,
        du=du,
        dose=dose,
        period=period,
        width=width,
        start=start,
    )
    interactions = [
        RateInteraction(
            'tumour_growth',
            target='tumour',
            state='A',
            rate=lambda s: -s.target.lambda1 * s.target.A * log(s.target.A / s.target.Vb),
        ),
        ReactionInteraction(
            'vessel_loss',
            target='tumour',
            reactants={'Vb': 1},
            propensity=lambda s: s.target.lambda2 * s.target.Vb,
        ),
        RateInteraction('stimulation', target='tumour', state='Vb', rate=lambda s: s.target.xi_s * s.target.A),
        ReactionInteraction(
            'inhibition',
            target='tumour',
            reactants={'Vb': 1},
            propensity=lambda s: s.target.xi_i * s.target.Vb * s.target.A ** (2 / 3),
        ),
        ReactionInteraction(
            'pruning',
            target='tumour',
            reactants={'Vb': 1},
            propensity=lambda s: s.target.a * s.target.Vb * s.target.U,
        ),
        DosingInteraction(
            'dosing',
            target='tumour',
            state='U',
            dose=lambda s: s.target.dose,
            period=tumour.period,
            width=tumour.width,
            start=tumour.start,
        ),
        ReactionInteraction(
            'clearance',
            target='tumour',
            reactants={'U': 1},
            propensity=lambda s: s.target.du * s.target.U,
        ),
    ]
    return System(elements=[tumour], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('anti-angiogenic therapy', 'ode', 'angiogenesis', 'necrosis', 'dosing'),
    references=(BOOK_REFERENCE, 'Csercsik et al. 2017'),
    defaults={'until': 60.0, 'dt': 0.2, 'solver': ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-8)},
)
def vascular_core_periphery(
    *,
    Vx: float = 0.005,
    xi_e: float = 0.8,
    du: float = INHIBITOR_CLEARANCE,
    dose: float = 5.0,
    period: float = 7.0,
    width: float = 1.0,
    start: float = 0.0,
) -> System:
    """Return a tumour with a live core, a live periphery, and a necrotic core, each fed by its own vasculature.

    The radius grows once the periphery is dense, cells pass from the periphery to the core as it grows, and core cells
    die when the core vasculature is too sparse. Starved cells release angiogenic factor, which grows both vascular
    compartments, and the inhibitor lowers the angiogenesis by `exp(-xi_e U)`. The book writes the radius, core
    volume, and tumour volume as increments. They are read as rates, so the increment of the radius is `dAr/dt` and
    the others follow from it per unit time, which is how the bi-compartmental model of Csercsik et al. advances. The
    book does not give the density `Vx` of the surrounding vasculature, which is set to the reference vascular fraction
    `rv_ref`, nor a schedule, which is a one-day infusion of 5 mg/kg/day every week. The other parameters and the
    initial state are the book's, the clearance `du` is `log(2) / 3.9` per day, lengths are in mm, volumes in mm³,
    time in days, and the inhibitor in mg/kg.

    Args:
        Vx:
            The vascular density of the surrounding tissue.

        xi_e:
            The efficiency of the inhibitor.

        du:
            The clearance rate of the inhibitor.

        dose:
            The infusion rate during a dose, 0 for no treatment.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Returns:
        system:
            The model, with the radius `tumour.Ar`, the cells `tumour.Ac`, `tumour.Ap`, and `tumour.An`, the vascular
            volumes `tumour.Vbc` and `tumour.Vbp`, and the inhibitor `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import vascular_core_periphery
        >>> vascular_core_periphery(dose=0.0).elements[0].dose
        0.0
    """
    tumour = VascularizedSpheroidElement(
        'tumour',
        Vx=Vx,
        xi_e=xi_e,
        du=du,
        dose=dose,
        period=period,
        width=width,
        start=start,
    )
    return System(elements=[tumour])
