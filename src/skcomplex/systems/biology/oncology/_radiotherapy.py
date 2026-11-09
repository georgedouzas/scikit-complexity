"""Oncology systems: tumours under radiotherapy and radiosensitizers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import CrossingInteraction, RateInteraction, ReactionInteraction
from ....core.components.laws.utils import periodic_guard, pulse_train
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


class IrradiatedTumourElement(Element):
    """Tumour in six compartments under radiation and a radiosensitizer.

    The radiation is given in linear-quadratic fractions.

    The proliferating cells follow `dA1/dt = r A1 exp(-a_gamma u_tot)`, where `u_tot` is the dose that the fractions of
    an irradiation accumulate. The lethally irradiated cells of `A5` divide once into `A6`, `dA5/dt = -(r + m1) A5` and
    `dA6/dt = 2 r A5 - m1 A6`, and both feed the dying chain `A2 -> A3 -> A4`, which leaves the tumour at the rate
    `m1`. The radiosensitizer is eliminated as `dUs/dt = -du Us`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `LQRadiosensitizerSystem` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A1:
            The initial volume of proliferating cells.

        A2:
            The initial volume of the first dying compartment.

        A3:
            The initial volume of the second dying compartment.

        A4:
            The initial volume of the third dying compartment.

        A5:
            The initial volume of lethally irradiated cells that divide once more.

        A6:
            The initial volume of lethally irradiated cells that no longer divide.

        Us:
            The initial plasma concentration of the radiosensitizer.

        u_tot:
            The initial accumulated dose.

        r:
            The growth rate.

        m1:
            The transit rate of the damaged cells.

        a_gamma:
            The inhibition of proliferation per unit of accumulated dose.

        du:
            The elimination rate of the radiosensitizer.

    Examples:
        >>> from skcomplex.systems.biology.oncology import IrradiatedTumourElement
        >>> IrradiatedTumourElement('tumour', A1=10.0).A1
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 27.0,
        A2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A3: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A4: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A5: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        A6: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Us: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u_tot: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.04,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.26,
        a_gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.004,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.544,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.A5 = A5
        self.A6 = A6
        self.Us = Us
        self.u_tot = u_tot
        self.r = r
        self.m1 = m1
        self.a_gamma = a_gamma
        self.du = du


class RadiationCompetitionElement(Element):
    """Normal and tumour cells in Lotka-Volterra competition under irradiation.

    The irradiation is periodic.

    The normal cells follow `dN/dt = r1 N (1 - b1 N) - c1 N A - epsilon N` and the tumour cells
    `dA/dt = r2 A (1 - b2 A) - c2 N A - a2 D(t) A`, where the irradiation `D` is 1 for `width` time units every `period`
    and 0 otherwise, and `epsilon N` is the radiation that spills over to the normal tissue.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `RadiotherapyCompetitionSystem` read its parameters.

    Args:
        label:
            The label of the tissue.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial concentration of normal cells.

        A:
            The initial concentration of tumour cells.

        r1:
            The growth rate of the normal cells.

        r2:
            The growth rate of the tumour cells.

        b1:
            The reciprocal carrying capacity of the normal cells.

        b2:
            The reciprocal carrying capacity of the tumour cells.

        c1:
            The rate at which the tumour cells suppress the normal cells.

        c2:
            The rate at which the normal cells suppress the tumour cells.

        a2:
            The per capita rate at which the radiation kills tumour cells during an irradiation.

        epsilon:
            The per capita rate at which the spilled radiation kills normal cells.

        period:
            The time between the starts of irradiations.

        width:
            The duration of an irradiation.

    Examples:
        >>> from skcomplex.systems.biology.oncology import RadiationCompetitionElement
        >>> RadiationCompetitionElement('tissue', N=0.5, A=0.8).A
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.008,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        period: float = 7.0,
        width: float = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.A = A
        self.r1 = r1
        self.r2 = r2
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.a2 = a2
        self.epsilon = epsilon
        self.period = period
        self.width = width


class IrradiationInteraction(CrossingInteraction):
    """Radiation fractions of a dose `d`.

    They move the lethally irradiated part of the proliferating cells.

    A fraction at the time `start + i period` moves the part `M = 1 - exp(-(1 + ap Us)(alpha d + beta d^2))` of the
    proliferating cells `A1` to the lethally irradiated cells `A5` and raises the accumulated dose `u_tot` by
    `(1 + au Us) d`, where `Us` is the radiosensitizer. Every fraction is an event labelled by the interaction.

    Args:
        label:
            The label of the interaction and of its events.

        target:
            The path of the tumour, with the states `A1`, `A5`, `Us`, and `u_tot`.

        alpha:
            The linear coefficient of the linear-quadratic system.

        beta:
            The quadratic coefficient of the linear-quadratic system.

        ap:
            The potency of the radiosensitizer in raising the cell kill.

        au:
            The potency of the radiosensitizer in raising the accumulated dose.

        dose:
            The dose of a fraction.

        period:
            The time between fractions.

        start:
            The time of the first fraction.

        fractions:
            The number of fractions.

    Raises:
        ValidationError:
            The period is not positive, or the number of fractions is not a positive integer.

    Examples:
        >>> from skcomplex.systems.biology.oncology import IrradiationInteraction
        >>> IrradiationInteraction('fraction', target='tumour', fractions=5).fractions
        5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.054,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0054,
        ap: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.42,
        au: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        period: float = 1.0,
        start: float = 0.125,
        fractions: int = 10,
    ) -> None:
        super().__init__(label, target=target, guard=self._get_guard, reset=self._get_resets())
        self.alpha = alpha
        self.beta = beta
        self.ap = ap
        self.au = au
        self.dose = dose
        self.period = period
        self.start = start
        self.fractions = fractions

    def _get_guard(self, namespace: SimpleNamespace) -> Expression:
        """Return the guard that crosses zero at every fraction."""
        return periodic_guard(namespace.t, self.period, self.start, self.fractions)

    def _get_damage(self, s: SimpleNamespace) -> Expression:
        """Return the linear-quadratic damage of a fraction.

        The damage is `(1 + ap Us)(alpha d + beta d^2)`.
        """
        return (1 + s.ap * s.target.Us) * (s.alpha * s.dose + s.beta * s.dose**2)

    def _get_resets(self) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the surviving and lethally irradiated cells, and the accumulated dose.

        The surviving cells are the proliferating ones.
        """
        return {
            'A1': lambda s: s.target.A1 * exp(-self._get_damage(s)),
            'A5': lambda s: s.target.A5 + (1 - exp(-self._get_damage(s))) * s.target.A1,
            'u_tot': lambda s: s.target.u_tot + (1 + s.au * s.target.Us) * s.dose,
        }


class RadiotherapyCompetitionSystem(BaseSystem):
    """Normal and tumour cells in Lotka-Volterra competition.

    The tumour is irradiated periodically.

    The tumour cells outcompete the normal cells without treatment. The irradiation kills tumour cells at the per capita
    rate `a2` for `width` days every `period` days, which takes the tumour below the level at which the normal cells can
    invade, so the two coexist, and the radiation that spills over removes normal cells at the rate `epsilon`. The book
    combines a periodic constant dose with the dose proportional to the tumour. A constant dose drives an eradicated
    tumour negative, so the periodic dose here is proportional to the tumour. The book gives no schedule, so the
    defaults irradiate five days a week. The concentrations are normalized by the carrying capacities, `b1 = b2 = 1`,
    and time is in days. The system has the normal cells `tissue.N` and the tumour cells `tissue.A`.

    Args:
        N:
            The initial concentration of normal cells.

        A:
            The initial concentration of tumour cells.

        r1:
            The growth rate of the normal cells.

        r2:
            The growth rate of the tumour cells.

        c1:
            The rate at which the tumour cells suppress the normal cells.

        c2:
            The rate at which the normal cells suppress the tumour cells.

        a2:
            The per capita kill rate of the radiation during an irradiation, between 0.01 and 0.2.

        epsilon:
            The per capita kill rate of the radiation that spills over to the normal cells.

        period:
            The time between the starts of irradiations.

        width:
            The duration of an irradiation.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import RadiotherapyCompetitionSystem
        >>> RadiotherapyCompetitionSystem(a2=0.1).prepare().system_.elements[0].a2
        0.1
    """

    field = 'biology/oncology'
    topics = ('radiotherapy', 'ode', 'competition', 'dosing')
    references = (BOOK, 'Belostotski and Freedman 2005', 'Liu and Yang 2014')
    until = 1400.0
    dt = 5.0

    def __init__(
        self,
        N: float = 0.5,
        A: float = 0.8,
        *,
        r1: float = 0.008,
        r2: float = 0.4,
        c1: float = 0.01,
        c2: float = 0.15,
        a2: float = 0.2,
        epsilon: float = 0.001,
        period: float = 7.0,
        width: float = 5.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.N = N
        self.A = A
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2
        self.a2 = a2
        self.epsilon = epsilon
        self.period = period
        self.width = width

    def _build_system(self) -> System:
        """Return the parts of normal and tumour cells in Lotka-Volterra competition."""
        tissue = RadiationCompetitionElement(
            'tissue',
            N=self.N,
            A=self.A,
            r1=self.r1,
            r2=self.r2,
            c1=self.c1,
            c2=self.c2,
            a2=self.a2,
            epsilon=self.epsilon,
            period=self.period,
            width=self.width,
        )
        interactions = [
            RateInteraction(
                'normal_growth',
                target='tissue',
                state='N',
                rate=lambda s: s.target.r1 * s.target.N * (1 - s.target.b1 * s.target.N),
            ),
            ReactionInteraction(
                'normal_competition',
                target='tissue',
                reactants={'N': 1},
                modifiers={'A': 1},
                propensity=lambda s: s.target.c1 * s.target.N * s.target.A,
            ),
            ReactionInteraction(
                'spillover',
                target='tissue',
                reactants={'N': 1},
                propensity=lambda s: s.target.epsilon * s.target.N,
            ),
            RateInteraction(
                'tumour_growth',
                target='tissue',
                state='A',
                rate=lambda s: s.target.r2 * s.target.A * (1 - s.target.b2 * s.target.A),
            ),
            ReactionInteraction(
                'tumour_competition',
                target='tissue',
                reactants={'A': 1},
                modifiers={'N': 1},
                propensity=lambda s: s.target.c2 * s.target.N * s.target.A,
            ),
            ReactionInteraction(
                'irradiation',
                target='tissue',
                reactants={'A': 1},
                propensity=lambda s: s.target.a2 * pulse_train(s.t, self.period, self.width) * s.target.A,
            ),
        ]
        return System(self.label, elements=[tissue], interactions=interactions)


class LQRadiosensitizerSystem(BaseSystem):
    """A xenograft tumour irradiated in fractions, with a radiosensitizer.

    The linear-quadratic system scores the fractions.

    Every fraction kills the part `1 - exp(-(1 + ap Us)(alpha d + beta d^2))` of the proliferating cells at once, which
    divide once more and die through a chain of damaged compartments, and the accumulated dose slows the growth of the
    survivors as `exp(-a_gamma u_tot)`. The radiosensitizer, given once at time 0, raises both effects while it lasts.
    Volumes are in mm³, doses in Gy, concentrations in μg/mL, and time in days, so the book's `alpha = 54` and `a_gamma
    = 4` per kGy are 0.054 and 0.004 per Gy, `beta = alpha / 10` per Gy², and the elimination rate of 0.231 per hour is
    5.544 per day. Fractions are instantaneous, so the dose-protraction factor is 1. The book's equations 6.10 and 6.13
    also move `m1 A1` from the proliferating cells to the dying chain, which would make the untreated tumour shrink at
    `r - m1 < 0`, so the chain is fed by the irradiated compartments alone and the tumour starts with no damaged cells.
    The book derives `Us(0)` from a dose without its bioavailability and volume, so the initial concentration is given
    directly. The system has the compartments `tumour.A1` to `tumour.A6`, the radiosensitizer `tumour.Us`, the
    accumulated dose `tumour.u_tot`, and the fractions as `fraction` events.

    Args:
        A1:
            The initial tumour volume, all proliferating.

        Us:
            The initial plasma concentration of the radiosensitizer.

        r:
            The growth rate.

        m1:
            The transit rate of the damaged cells.

        a_gamma:
            The inhibition of proliferation per unit of accumulated dose.

        alpha:
            The linear coefficient of the linear-quadratic system.

        beta:
            The quadratic coefficient of the linear-quadratic system.

        ap:
            The potency of the radiosensitizer in raising the cell kill.

        au:
            The potency of the radiosensitizer in raising the accumulated dose.

        du:
            The elimination rate of the radiosensitizer.

        dose:
            The dose of a fraction.

        period:
            The time between fractions.

        start:
            The time of the first fraction.

        fractions:
            The number of fractions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import LQRadiosensitizerSystem
        >>> LQRadiosensitizerSystem(Us=0.0).prepare().system_.elements[0].Us
        0.0
    """

    field = 'biology/oncology'
    topics = ('radiotherapy', 'ode', 'linear-quadratic', 'events', 'dosing')
    references = (BOOK, 'Cardilin et al. 2019')
    until = 120.0
    dt = 0.5

    def __init__(
        self,
        A1: float = 27.0,
        Us: float = 2.0,
        *,
        r: float = 0.04,
        m1: float = 0.26,
        a_gamma: float = 0.004,
        alpha: float = 0.054,
        beta: float = 0.0054,
        ap: float = 0.42,
        au: float = 0.15,
        du: float = 5.544,
        dose: float = 2.0,
        period: float = 1.0,
        start: float = 0.125,
        fractions: int = 10,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A1 = A1
        self.Us = Us
        self.r = r
        self.m1 = m1
        self.a_gamma = a_gamma
        self.alpha = alpha
        self.beta = beta
        self.ap = ap
        self.au = au
        self.du = du
        self.dose = dose
        self.period = period
        self.start = start
        self.fractions = fractions

    def _build_system(self) -> System:
        """Return the parts of a xenograft tumour irradiated in fractions.

        The linear-quadratic system scores the fractions.
        """
        tumour = IrradiatedTumourElement(
            'tumour',
            A1=self.A1,
            Us=self.Us,
            r=self.r,
            m1=self.m1,
            a_gamma=self.a_gamma,
            du=self.du,
        )
        irradiation = IrradiationInteraction(
            'fraction',
            target='tumour',
            alpha=self.alpha,
            beta=self.beta,
            ap=self.ap,
            au=self.au,
            dose=self.dose,
            period=self.period,
            start=self.start,
            fractions=self.fractions,
        )
        interactions = [
            RateInteraction(
                'growth',
                target='tumour',
                state='A1',
                rate=lambda s: s.target.r * s.target.A1 * exp(-s.target.a_gamma * s.target.u_tot),
            ),
            ReactionInteraction(
                'damaged_division',
                target='tumour',
                reactants={'A5': 1},
                products={'A6': 2},
                propensity=lambda s: s.target.r * s.target.A5,
            ),
            ReactionInteraction(
                'arrest',
                target='tumour',
                reactants={'A5': 1},
                products={'A2': 1},
                propensity=lambda s: s.target.m1 * s.target.A5,
            ),
            ReactionInteraction(
                'daughter_arrest',
                target='tumour',
                reactants={'A6': 1},
                products={'A2': 1},
                propensity=lambda s: s.target.m1 * s.target.A6,
            ),
            ReactionInteraction(
                'first_dying_transit',
                target='tumour',
                reactants={'A2': 1},
                products={'A3': 1},
                propensity=lambda s: s.target.m1 * s.target.A2,
            ),
            ReactionInteraction(
                'second_dying_transit',
                target='tumour',
                reactants={'A3': 1},
                products={'A4': 1},
                propensity=lambda s: s.target.m1 * s.target.A3,
            ),
            ReactionInteraction(
                'removal',
                target='tumour',
                reactants={'A4': 1},
                propensity=lambda s: s.target.m1 * s.target.A4,
            ),
            ReactionInteraction(
                'sensitizer_elimination',
                target='tumour',
                reactants={'Us': 1},
                propensity=lambda s: s.target.du * s.target.Us,
            ),
            irradiation,
        ]
        return System(self.label, elements=[tumour], interactions=interactions)
