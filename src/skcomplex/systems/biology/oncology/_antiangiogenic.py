"""Oncology systems: tumours whose vasculature an angiogenesis inhibitor prunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, log, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import logistic, pulse_train
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ...general.reactions import ReactionInteraction
from ..common.pharmacokinetics import DosingInteraction

INHIBITOR_CLEARANCE = math.log(2) / 3.9
BOOK_REFERENCE = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class AngiogenicTumourElement(Element):
    """Tumour volume, its vascular volume, and an angiogenesis inhibitor.

    The vascular volume sets the carrying capacity of the tumour.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)`, the vasculature
    `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb A^(2/3) - a Vb U`, and the inhibitor `dU/dt = -du U + u(t)`, where the
    infusion `u` is `dose` for `width` time units every `period`, from `start` on. A `width` equal to the `period`
    is a continuous infusion.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `HahnfeldtSystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial tumour volume.

        Vb:
            The initial vascular volume.

        U:
            The initial concentration of the inhibitor.

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
            The infusion rate during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Examples:
        >>> from skcomplex.systems.biology.oncology import AngiogenicTumourElement
        >>> AngiogenicTumourElement('tumour').xi_s
        5.85
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 17347.0,
        Vb: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 17347.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lambda1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.192,
        lambda2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        xi_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.85,
        xi_i: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.00873,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.66,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        period: float = 1.0,
        width: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.Vb = Vb
        self.U = U
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.xi_s = xi_s
        self.xi_i = xi_i
        self.a = a
        self.du = du
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start


class HahnfeldtSystem(BaseSystem):
    """The Hahnfeldt system, a tumour whose carrying capacity is its vasculature.

    The tumour stimulates and inhibits the vasculature.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)` and the vasculature `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb
    A^(2/3) - a Vb U`. The inhibitor follows `dU/dt = -du U + u(t)`, the differential form of the book's convolution.
    Untreated, tumour and vasculature tend to the volume `(xi_s / xi_i)^(3/2)`, about 17347 mm³, which is the initial
    state. The inhibitor parameters are those of endostatin in Lewis lung carcinoma, and the default schedule is the
    continuous infusion of 20 mg/kg/day of Hahnfeldt et al., under which the vasculature and then the tumour regress.
    Volumes are in mm³, time in days, and the inhibitor in mg/kg. The system has the volumes `tumour.A` and `tumour.Vb`
    and the inhibitor `tumour.U`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import HahnfeldtSystem
        >>> HahnfeldtSystem(dose=0.0).prepare().system_.elements[0].dose
        0.0
    """

    field = 'biology/oncology'
    topics = ('anti-angiogenic therapy', 'ode', 'angiogenesis', 'Gompertz', 'dosing')
    references = (BOOK_REFERENCE, 'Hahnfeldt et al. 1999')
    until = 30.0
    dt = 0.1

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A = A
        self.Vb = Vb
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.xi_s = xi_s
        self.xi_i = xi_i
        self.a = a
        self.du = du
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _build_system(self) -> CompositeSystem:
        """Return the components of the Hahnfeldt system."""
        tumour = AngiogenicTumourElement(
            'tumour',
            A=self.A,
            Vb=self.Vb,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            xi_s=self.xi_s,
            xi_i=self.xi_i,
            a=self.a,
            du=self.du,
            dose=self.dose,
            period=self.period,
            width=self.width,
            start=self.start,
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
        return CompositeSystem(self.label, elements=[tumour], interactions=interactions)


class VascularCorePeripherySystem(DynamicalSystem):
    """A tumour with a live core, a live periphery, and a necrotic core.

    Each compartment is fed by its own vasculature.

    The radius grows once the periphery is dense, cells pass from the periphery to the core as it grows, and core cells
    die when the core vasculature is too sparse. Starved cells release angiogenic factor, which grows both vascular
    compartments, and the inhibitor lowers the angiogenesis by `exp(-xi_e U)`. The book writes the radius, core volume,
    and tumour volume as increments. They are read as rates, so the increment of the radius is `dAr/dt` and the others
    follow from it per unit time, which is how the bi-compartmental system of Csercsik et al. advances. The book does
    not give the density `Vx` of the surrounding vasculature, which is set to the reference vascular fraction `rv_ref`,
    nor a schedule, which is a one-day infusion of 5 mg/kg/day every week. The other parameters and the initial state
    are the book's, the clearance `du` is `log(2) / 3.9` per day, lengths are in mm, volumes in mm³, time in days, and
    the inhibitor in mg/kg. The system has the radius `tumour.Ar`, the cells `tumour.Ac`, `tumour.Ap`, and `tumour.An`,
    the vascular volumes `tumour.Vbc` and `tumour.Vbp`, and the inhibitor `tumour.U`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import VascularCorePeripherySystem
        >>> VascularCorePeripherySystem(dose=0.0).prepare().system_.elements[0].get_value('dose')
        0.0
    """

    field = 'biology/oncology'
    topics = ('anti-angiogenic therapy', 'ode', 'angiogenesis', 'necrosis', 'dosing')
    references = (BOOK_REFERENCE, 'Csercsik et al. 2017')
    until = 60.0
    dt = 0.2
    solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-8)

    def __init__(
        self,
        *,
        Vx: float = 0.005,
        xi_e: float = 0.8,
        du: float = INHIBITOR_CLEARANCE,
        dose: float = 5.0,
        period: float = 7.0,
        width: float = 1.0,
        start: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.Vx = Vx
        self.xi_e = xi_e
        self.du = du
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'tumour'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'Ar': DeclaredVariableSpec(13.0, StateSpec(domain=RealSet.open(0, oo))),
            'Ac': DeclaredVariableSpec(3.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'Ap': DeclaredVariableSpec(495.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'An': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'Vbc': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'Vbp': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'U': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'r1': DeclaredVariableSpec(1.2, ParameterSpec(domain=RealSet.open(0, oo))),
            'r2': DeclaredVariableSpec(12.0, ParameterSpec(domain=RealSet.open(0, oo))),
            'r3': DeclaredVariableSpec(0.06, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'm': DeclaredVariableSpec(1.0, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'a1': DeclaredVariableSpec(1.0, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'p1': DeclaredVariableSpec(1.0, ParameterSpec()),
            'p2': DeclaredVariableSpec(400000.0, ParameterSpec()),
            'p3': DeclaredVariableSpec(0.2, ParameterSpec()),
            'p4': DeclaredVariableSpec(0.45, ParameterSpec()),
            'p_g': DeclaredVariableSpec(400000.0, ParameterSpec()),
            'kappa1': DeclaredVariableSpec(0.17, ParameterSpec()),
            'kappa2': DeclaredVariableSpec(-45000.0, ParameterSpec()),
            'kappa3': DeclaredVariableSpec(-0.04, ParameterSpec()),
            'kappa4': DeclaredVariableSpec(-0.06, ParameterSpec()),
            'kappa_g': DeclaredVariableSpec(55000.0, ParameterSpec()),
            'theta1': DeclaredVariableSpec(0.3, ParameterSpec(domain=RealSet.open(0, oo))),
            'theta2': DeclaredVariableSpec(0.3, ParameterSpec(domain=RealSet.open(0, oo))),
            'beta': DeclaredVariableSpec(1.2, ParameterSpec(domain=RealSet.open(0, oo))),
            'xi_e': DeclaredVariableSpec(self.xi_e, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'rv_ref': DeclaredVariableSpec(0.005, ParameterSpec(domain=RealSet.open(0, oo))),
            'Vx': DeclaredVariableSpec(self.Vx, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'du': DeclaredVariableSpec(self.du, ParameterSpec(domain=RealSet.open(0, oo))),
            'dose': DeclaredVariableSpec(self.dose, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the growth of the radius, the balances, and the inhibitor kinetics."""
        return {
            'Ar': self._get_growth,
            'Ac': lambda s: self._transfer(s) * s.target.Ap - self._get_necrosis(s),
            'Ap': lambda s: -self._transfer(s) * s.target.Ap + self._get_proliferation(s),
            'An': self._get_necrosis,
            'Vbc': lambda s: self._get_internalization(s) + self._get_angiogenesis(s),
            'Vbp': lambda s: s.target.Vx * self._get_volume_increment(s)
            - self._get_internalization(s)
            + self._get_angiogenesis(s),
            'U': lambda s: self._get_infusion(s) - s.target.du * s.target.U,
        }

    def _get_ratio(self, s: SimpleNamespace) -> Expression:
        """Return `ratio` from the namespace of an equation."""
        return VascularCorePeripherySystem._get_core_ratio(s.target.Ar)

    def _get_core(self, s: SimpleNamespace) -> Expression:
        """Return `core` from the namespace of an equation."""
        return VascularCorePeripherySystem._get_ball(self._get_ratio(s) * s.target.Ar)

    def _get_periphery(self, s: SimpleNamespace) -> Expression:
        """Return `periphery` from the namespace of an equation."""
        return VascularCorePeripherySystem._get_ball(s.target.Ar) - self._get_core(s)

    def _get_density(self, s: SimpleNamespace) -> Expression:
        """Return `density` from the namespace of an equation."""
        return s.target.Ap / self._get_periphery(s)

    def _get_growth(self, s: SimpleNamespace) -> Expression:
        """Return `growth` from the namespace of an equation."""
        return s.target.r1 * logistic(self._get_density(s), s.target.p_g, s.target.kappa_g)

    def _get_core_increment(self, s: SimpleNamespace) -> Expression:
        """Return `core_increment` from the namespace of an equation."""
        return VascularCorePeripherySystem._get_ball(
            self._get_ratio(s) * (s.target.Ar + self._get_growth(s)),
        ) - self._get_core(s)

    def _get_volume_increment(self, s: SimpleNamespace) -> Expression:
        """Return `volume_increment` from the namespace of an equation."""
        return VascularCorePeripherySystem._get_ball(
            s.target.Ar + self._get_growth(s),
        ) - VascularCorePeripherySystem._get_ball(
            s.target.Ar,
        )

    def _transfer(self, s: SimpleNamespace) -> Expression:
        """Return `transfer` from the namespace of an equation."""
        return self._get_core_increment(s) / self._get_periphery(s)

    def _get_core_nutrient(self, s: SimpleNamespace) -> Expression:
        """Return `core_nutrient` from the namespace of an equation."""
        return s.target.Vbc / (self._get_core(s) * s.target.rv_ref)

    def _get_periphery_nutrient(self, s: SimpleNamespace) -> Expression:
        """Return `periphery_nutrient` from the namespace of an equation."""
        return s.target.Vbp / (self._get_periphery(s) * s.target.rv_ref)

    def _get_necrosis(self, s: SimpleNamespace) -> Expression:
        """Return `necrosis` from the namespace of an equation."""
        return (
            s.target.m
            * logistic(self._get_core_nutrient(s), s.target.p3, s.target.kappa3) ** s.target.theta1
            * s.target.Ac
        )

    def _get_proliferation(self, s: SimpleNamespace) -> Expression:
        """Return `proliferation` from the namespace of an equation."""
        return (
            s.target.r2
            * logistic(self._get_periphery_nutrient(s), s.target.p1, s.target.kappa1)
            * logistic(self._get_density(s), s.target.p2, s.target.kappa2)
            * s.target.Ap
        )

    def _get_signal(self, s: SimpleNamespace) -> Expression:
        """Return `signal` from the namespace of an equation."""
        return logistic(self._get_core_nutrient(s), s.target.p4, s.target.kappa4) ** s.target.theta2

    def _get_angiogenesis(self, s: SimpleNamespace) -> Expression:
        """Return `angiogenesis` from the namespace of an equation."""
        return s.target.a1 * exp(-s.target.xi_e * s.target.U) * self._get_signal(s) * s.target.Vbp

    def _get_internalization(self, s: SimpleNamespace) -> Expression:
        """Return `internalization` from the namespace of an equation."""
        return s.target.r3 * s.target.Ar**s.target.beta * self._transfer(s) * s.target.Vbp

    def _get_infusion(self, s: SimpleNamespace) -> Expression:
        """Return `infusion` from the namespace of an equation."""
        return s.target.dose * pulse_train(s.t, self.period, self.width, start=self.start)

    @staticmethod
    def _get_ball(radius: Expression) -> Expression:
        """Return the volume of a ball of a radius."""
        return 4 * pi * radius**3 / 3

    @staticmethod
    def _get_core_ratio(radius: Expression) -> Expression:
        """Return the core-periphery ratio of a tumour radius.

        The ratio is `(1 / (1 + exp((0.5 - Ar) / 0.333)))^2.2 - 0.15`.
        """
        return logistic(radius, 0.5, 0.333) ** 2.2 - 0.15
