"""Physiology models: the distribution and elimination of a drug between two compartments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System
from ....core.runtime import register_model


class TwoCompartment(Element):
    """Drug in a central compartment, where it is eliminated, exchanged with a peripheral compartment.

    The central concentration follows `dUc/dt = k21 Up - (k10 + k12) Uc` and the peripheral one
    `dUp/dt = k12 Uc - k21 Up`.

    Args:
        label:
            The label of the patient.

        Uc:
            The initial central concentration.

        Up:
            The initial peripheral concentration.

        k10:
            The elimination rate from the central compartment.

        k12:
            The transfer rate from the central to the peripheral compartment.

        k21:
            The transfer rate from the peripheral to the central compartment.

    Examples:
        >>> from skcomplex.models.biology.physiology import TwoCompartment
        >>> TwoCompartment('patient').k12
        0.0475
    """

    Uc: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    Up: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    k10: float = parameter(default=0.082, domain=Domain.NONNEGATIVE)
    k12: float = parameter(default=0.0475, domain=Domain.NONNEGATIVE)
    k21: float = parameter(default=0.00125, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the exchange between the compartments and the elimination from the central one.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        transfer = s.k12 * s.Uc - s.k21 * s.Up
        return [Rate(s.Uc, -transfer - s.k10 * s.Uc), Rate(s.Up, transfer)]


@register_model(
    field='biology/physiology',
    tags=('pharmacology', 'pharmacokinetics', 'ode', 'compartments'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Hendriks et al. 2012'),
    defaults={'until': 3000.0, 'dt': 10.0},
)
def two_compartment_pharmacokinetics(
    c0: float = 1.0,
    *,
    k10: float = 0.082,
    k12: float = 0.0475,
    k21: float = 0.00125,
) -> System:
    """Return the two-compartment pharmacokinetics of a bolus dose given into the central compartment.

    The central concentration follows `dUc/dt = k21 Up - (k10 + k12) Uc` and the peripheral one
    `dUp/dt = k12 Uc - k21 Up`, from `Uc = c0` and `Up = 0`. The central concentration is the bi-exponential
    `c0 ((alpha - k21) e^(-alpha t) + (k21 - beta) e^(-beta t)) / (alpha - beta)`, where `alpha` and `beta` are the
    roots of `x^2 - (k10 + k12 + k21) x + k10 k21`: a fast distribution phase is followed by a slow elimination
    phase. The total drug `Uc + Up` falls only by the elimination `k10 Uc`. The rates are those of doxorubicin, per
    minute.

    Args:
        c0:
            The bolus, the initial central concentration.

        k10:
            The elimination rate from the central compartment, per minute.

        k12:
            The transfer rate from the central to the peripheral compartment, per minute.

        k21:
            The transfer rate from the peripheral to the central compartment, per minute.

    Returns:
        system:
            The model, with the central concentration `patient.Uc` and the peripheral one `patient.Up`.

    Examples:
        >>> from skcomplex.models.biology.physiology import two_compartment_pharmacokinetics
        >>> two_compartment_pharmacokinetics(c0=5.0).elements[0].Uc
        5.0
    """
    return System(elements=[TwoCompartment('patient', Uc=c0, k10=k10, k12=k12, k21=k21)])
