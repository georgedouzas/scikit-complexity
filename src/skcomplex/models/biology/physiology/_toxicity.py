"""Physiology models: the toxicity a drug accumulates in the body."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System, pulse_train, step
from ....core.runtime import register_model


class DrugToxicity(Element):
    """Drug in the plasma, given in a train of infusions and eliminated, and the toxicity it accumulates.

    The plasma concentration follows `dc/dt = u(t) - k c`, where the infusion `u` is `dose` for `width` time units
    every `period`, and the toxicity `dT/dt = c - eta T`, which the metabolism lowers at the rate `eta`.

    Args:
        label:
            The label of the patient.

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
        >>> from skcomplex.models.biology.physiology import DrugToxicity
        >>> DrugToxicity('patient', doses=4).doses
        4
    """

    c: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    T: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    k: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    eta: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    dose: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    period: float = 7.0
    width: float = 1.0
    doses: int = 6

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balance of the drug and the accumulation of toxicity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        infusion = pulse_train(s.t, self.period, self.width) * (1 - step(s.t, self.doses * self.period))
        return [Rate(s.c, s.dose * infusion - s.k * s.c), Rate(s.T, s.c - s.eta * s.T)]


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
    patient = DrugToxicity('patient', k=k, eta=eta, dose=dose, period=period, width=width, doses=doses)
    return System(elements=[patient])
