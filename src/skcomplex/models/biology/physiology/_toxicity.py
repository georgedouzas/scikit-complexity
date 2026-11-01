"""Physiology models: the toxicity a drug accumulates in the body."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import RateInteraction, ReactionInteraction
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.pharmacokinetics import DosingInteraction


class DrugToxicityElement(Element):
    """Drug in the plasma, given in a train of infusions and eliminated, and the toxicity it accumulates.

    The plasma concentration follows `dc/dt = u(t) - k c`, where the infusion `u` is `dose` for `width` time units
    every `period`, and the toxicity `dT/dt = c - eta T`, which the metabolism lowers at the rate `eta`.

    The element holds the states and parameters and has no laws of its own: the interactions of `drug_toxicity` read its
    parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The initial plasma concentration.

        T:
            The initial toxicity.

        k:
            The elimination rate of the drug.

        eta:
            The rate at which the metabolism lowers the toxicity, 0 for a cumulative toxicity.

        dose:
            The infusion rate during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.models.biology.physiology import DrugToxicityElement
        >>> DrugToxicityElement('patient', doses=4).doses
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        T: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        period: float = 7.0,
        width: float = 1.0,
        doses: int = 6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.T = T
        self.k = k
        self.eta = eta
        self.dose = dose
        self.period = period
        self.width = width
        self.doses = doses


@register_model(
    field='biology/physiology',
    tags=('pharmacology', 'toxicity', 'ode', 'dosing'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Martin and Teo 1994'),
    defaults={'until': 84.0, 'dt': 0.2},
)
def drug_toxicity(
    *,
    k: float = 0.5,
    eta: float = 0.05,
    dose: float = 1.0,
    period: float = 7.0,
    width: float = 1.0,
    doses: int = 6,
) -> System:
    """Return the toxicity a weekly drug accumulates in the body, which the metabolism clears once the doses stop.

    Every dose raises the plasma concentration, which the body eliminates within days, and the toxicity integrates
    the concentration, `dT/dt = c - eta T`. With `eta = 0` the toxicity is the cumulative exposure and never falls.

    Args:
        k:
            The elimination rate of the drug, per day.

        eta:
            The rate at which the metabolism lowers the toxicity, per day.

        dose:
            The infusion rate during a dose.

        period:
            The time between the starts of doses, in days.

        width:
            The duration of a dose, in days.

        doses:
            The number of doses.

    Returns:
        system:
            The model, with the concentration `patient.c` and the toxicity `patient.T`.

    Examples:
        >>> from skcomplex.models.biology.physiology import drug_toxicity
        >>> drug_toxicity(eta=0.0).elements[0].eta
        0.0
    """
    patient = DrugToxicityElement('patient', k=k, eta=eta, dose=dose, period=period, width=width, doses=doses)
    interactions = [
        DosingInteraction(
            'dosing',
            target='patient',
            state='c',
            dose=lambda s: s.target.dose,
            period=period,
            width=width,
            count=doses,
        ),
        ReactionInteraction(
            'clearance',
            target='patient',
            reactants={'c': 1},
            propensity=lambda s: s.target.k * s.target.c,
        ),
        RateInteraction('toxicity', target='patient', state='T', rate=lambda s: s.target.c),
        ReactionInteraction(
            'toxicity_recovery',
            target='patient',
            reactants={'T': 1},
            propensity=lambda s: s.target.eta * s.target.T,
        ),
    ]
    return System(elements=[patient], interactions=interactions)
