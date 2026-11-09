"""The distribution and elimination of a drug between two compartments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import ReactionInteraction
from ....core.components.systems import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec


class TwoCompartmentElement(Element):
    """Drug concentrations of a patient in two compartments.

    They are the central compartment `Uc` and the peripheral compartment `Up`.

    The compartments have no laws of their own. Reactions exchange the drug between them and eliminate it.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Uc:
            The initial central concentration.

        Up:
            The initial peripheral concentration.

    Examples:
        >>> from skcomplex.systems.biology.physiology import TwoCompartmentElement
        >>> TwoCompartmentElement('patient').Up
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Uc: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        Up: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Uc = Uc
        self.Up = Up


class TwoCompartmentPharmacokineticsSystem(BaseSystem):
    """The two-compartment pharmacokinetics of a bolus dose.

    The dose is given into the central compartment.

    The central concentration follows `dUc/dt = k21 Up - (k10 + k12) Uc` and the peripheral one `dUp/dt = k12 Uc - k21
    Up`, from `Uc = c0` and `Up = 0`. The central concentration is the bi-exponential `c0 ((alpha - k21) e^(-alpha t) +
    (k21 - beta) e^(-beta t)) / (alpha - beta)`, where `alpha` and `beta` are the roots of `x^2 - (k10 + k12 + k21) x +
    k10 k21`: a fast distribution phase is followed by a slow elimination phase. The total drug `Uc + Up` falls only by
    the elimination `k10 Uc`. The rates are those of doxorubicin, per minute. The system has the central concentration
    `patient.Uc` and the peripheral one `patient.Up`.

    Args:
        c0:
            The bolus, the initial central concentration.

        k10:
            The elimination rate from the central compartment, per minute.

        k12:
            The transfer rate from the central to the peripheral compartment, per minute.

        k21:
            The transfer rate from the peripheral to the central compartment, per minute.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import TwoCompartmentPharmacokineticsSystem
        >>> TwoCompartmentPharmacokineticsSystem(c0=5.0).prepare().system_.elements[0].Uc
        5.0
    """

    field = 'biology/physiology'
    topics = ('pharmacology', 'pharmacokinetics', 'ode', 'compartments')
    references = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Hendriks et al. 2012')
    until = 3000.0
    dt = 10.0

    def __init__(
        self,
        c0: float = 1.0,
        *,
        k10: float = 0.082,
        k12: float = 0.0475,
        k21: float = 0.00125,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.c0 = c0
        self.k10 = k10
        self.k12 = k12
        self.k21 = k21

    def _build_system(self) -> System:
        """Return the parts of the system."""
        reactions = [
            ReactionInteraction(
                'distribution',
                reactants={('patient', 'Uc'): 1},
                products={('patient', 'Up'): 1},
                k=self.k12,
            ),
            ReactionInteraction(
                'redistribution',
                reactants={('patient', 'Up'): 1},
                products={('patient', 'Uc'): 1},
                k=self.k21,
            ),
            ReactionInteraction('elimination', reactants={('patient', 'Uc'): 1}, k=self.k10),
        ]
        return System(self.label, elements=[TwoCompartmentElement('patient', Uc=self.c0)], interactions=reactions)
