"""Organization models: Meadows' system traps of success to the successful, shifting the burden, and escalation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.laws.utils import distributed_delay, lookup, where
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model

ALLOCATION_TABLE = ((0.0, 0.0), (0.2, 0.05), (0.5, 0.5), (0.8, 0.95), (1.0, 1.0))


def _success(s: SimpleNamespace) -> Expression:
    """Return the past success of the first competitor, its fraction of all resources."""
    return s.a / (s.a + s.b)


def _allocation(s: SimpleNamespace) -> Expression:
    """Return the share of the budget the first competitor gets."""
    return lookup(_success(s), ALLOCATION_TABLE)


def _contest_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of two competitors that share a budget in proportion to their past success."""
    allocation = _allocation(s)
    return [
        TransferLaw(Source('budget'), s.a, s.budget * allocation, label='allocation_a'),
        TransferLaw(Source('budget'), s.b, s.budget * (1 - allocation), label='allocation_b'),
        TransferLaw(s.a, Sink('depreciation'), s.a / s.lifetime, label='depreciation_a'),
        TransferLaw(s.b, Sink('depreciation'), s.b / s.lifetime, label='depreciation_b'),
    ]


class ContestElement(Element):
    """Contest element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='resources')] = 55.0,
        b: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='resources')] = 45.0,
        budget: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        lifetime: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.budget = budget
        self.lifetime = lifetime

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the contest element."""
        return _contest_laws(s)


def _symptom(s: SimpleNamespace) -> Expression:
    """Return the symptom, the gap between the desired and the current state of the problem."""
    return s.desired - s.capability - s.intervention


def _burden_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of a problem that a quick fix relieves while it erodes the capability of a fundamental one."""
    symptom = _symptom(s)
    building = distributed_delay(symptom, conserved=False, time=s.solution_delay) / s.build_time
    return [
        TransferLaw(Source('intervener'), s.intervention, s.fix_rate * symptom, label='fixing'),
        TransferLaw(Source('learning'), s.capability, building, label='building'),
        TransferLaw(s.capability, Sink('atrophy'), s.erosion * s.intervention * s.capability, label='erosion'),
    ]


class BurdenElement(Element):
    """Burden element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        capability: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='capability')] = 0.5,
        intervention: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='intervention')] = 0.0,
        desired: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        fix_rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        build_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        solution_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        erosion: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.capability = capability
        self.intervention = intervention
        self.desired = desired
        self.fix_rate = fix_rate
        self.build_time = build_time
        self.solution_delay = solution_delay
        self.erosion = erosion

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the burden element."""
        return _burden_laws(s)


def _desired_a(s: SimpleNamespace) -> Expression:
    """Return the stock the first party desires, a margin above its delayed view of the second's."""
    return (1 + s.margin) * distributed_delay(s.b, conserved=False, time=s.perception_delay)


def _desired_b(s: SimpleNamespace) -> Expression:
    """Return the stock the second party desires, a margin above its delayed view of the first's."""
    return (1 + s.margin) * distributed_delay(s.a, conserved=False, time=s.perception_delay)


def _escalation_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of two parties that each build their stock up to a margin above their view of the other's."""
    desired_a, desired_b = _desired_a(s), _desired_b(s)
    build_a = where(desired_a > s.a, desired_a - s.a, 0) / s.adjustment_time
    build_b = where(desired_b > s.b, desired_b - s.b, 0) / s.adjustment_time
    return [
        TransferLaw(Source('production'), s.a, build_a, label='build_a'),
        TransferLaw(Source('production'), s.b, build_b, label='build_b'),
    ]


class ArmsRaceElement(Element):
    """Arms race element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='arms')] = 10.0,
        b: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='arms')] = 10.5,
        margin: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        adjustment_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        perception_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.margin = margin
        self.adjustment_time = adjustment_time
        self.perception_delay = perception_delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the arms race element."""
        return _escalation_laws(s)


@register_model(
    field='social/organizations',
    tags=('system traps', 'competitive exclusion', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 0.5},
    observables=lambda system: [
        ObservableSpec('success', _success, element='contest'),
        ObservableSpec('allocation', _allocation, element='contest'),
    ],
)
def success_to_the_successful(
    a: float = 55.0,
    b: float = 45.0,
    *,
    budget: float = 10.0,
    lifetime: float = 10.0,
) -> System:
    """Return Meadows' success to the successful trap, two competitors that share a budget by their past success.

    Each competitor holds a stock of resources that its share of the budget fills and that depreciates over its
    lifetime. The share of the first competitor is a table function of its success, its fraction of all resources,
    that gives more than a proportional share to the more successful one. The two reinforcing loops make the symmetric
    state unstable, so the competitor that starts larger ends with the whole budget and the other one disappears. The
    book gives the structure only, so the table and the parameters are a documented choice.

    Args:
        a:
            The initial resources of the first competitor.

        b:
            The initial resources of the second competitor.

        budget:
            The resources allocated per year.

        lifetime:
            The average lifetime of resources, in years.

    Returns:
        system:
            The model, with the resources `contest.a` and `contest.b` in years, the flows `contest.allocation_a`,
            `contest.allocation_b`, `contest.depreciation_a`, and `contest.depreciation_b`. The observables of the
            model are `contest.success` and `contest.allocation`.

    Examples:
        >>> from skcomplex.models.social.organizations import success_to_the_successful
        >>> success_to_the_successful(a=60.0).elements[0].a
        60.0
    """
    return System(elements=[ContestElement('contest', a=a, b=b, budget=budget, lifetime=lifetime)])


@register_model(
    field='social/organizations',
    tags=('system traps', 'addiction', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 60.0, 'dt': 0.5},
    observables=lambda system: [ObservableSpec('symptom', _symptom, element='burden')],
)
def shifting_the_burden(
    capability: float = 0.5,
    intervention: float = 0.0,
    *,
    desired: float = 1.0,
    fix_rate: float = 1.0,
    build_time: float = 5.0,
    solution_delay: float = 2.0,
    erosion: float = 0.5,
) -> System:
    """Return Meadows' shifting the burden trap, a quick fix that relieves a symptom and erodes the system's capability.

    The symptom is the gap between the desired state and the state that the capability of the system and an outside
    intervention deliver together. The intervention grows quickly with the symptom. The capability grows slowly with
    the symptom, perceived over the solution delay, and the intervention erodes it as a side effect. With a quick fix
    the intervention takes over and the capability withers away, an addiction. Without one, `fix_rate = 0`, the
    capability grows to the desired state. The book gives the structure only, so the parameters are a documented
    choice.

    Args:
        capability:
            The initial capability of the system to deliver the desired state on its own.

        intervention:
            The initial level of the intervention.

        desired:
            The desired state.

        fix_rate:
            The rate at which the intervention grows per unit of symptom, 0 for no quick fix.

        build_time:
            The time the fundamental solution needs to close a perceived symptom.

        solution_delay:
            The time over which the fundamental solution perceives the symptom.

        erosion:
            The rate at which a unit of intervention erodes the capability.

    Returns:
        system:
            The model, with the capability `burden.capability` and the intervention `burden.intervention` in years,
            the flows `burden.fixing`, `burden.building`, and `burden.erosion`. The observable of the model is
            `burden.symptom`.

    Examples:
        >>> from skcomplex.models.social.organizations import shifting_the_burden
        >>> shifting_the_burden(fix_rate=0.0).elements[0].fix_rate
        0.0
    """
    burden = BurdenElement(
        'burden',
        capability=capability,
        intervention=intervention,
        desired=desired,
        fix_rate=fix_rate,
        build_time=build_time,
        solution_delay=solution_delay,
        erosion=erosion,
    )
    return System(elements=[burden])


@register_model(
    field='social/organizations',
    tags=('system traps', 'arms race', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 0.5},
    observables=lambda system: [
        ObservableSpec('desired_a', _desired_a, element='race'),
        ObservableSpec('desired_b', _desired_b, element='race'),
    ],
)
def escalation(
    a: float = 10.0,
    b: float = 10.5,
    *,
    margin: float = 0.1,
    adjustment_time: float = 2.0,
    perception_delay: float = 1.0,
) -> System:
    """Return Meadows' escalation trap, two parties that each want to exceed the other's stock by a margin.

    Each party sees the stock of the other through an information smoothing over the perception delay, wants to hold
    `1 + margin` times that stock, and builds up the gap over the adjustment time. The loop through both stocks is
    reinforcing, so both stocks grow exponentially at the rate `r` that solves
    `r * adjustment_time = (1 + margin) / (1 + r * perception_delay) - 1`. The book gives the structure only, so the
    parameters are a documented choice.

    Args:
        a:
            The initial stock of the first party.

        b:
            The initial stock of the second party.

        margin:
            The fraction by which each party wants to exceed the other.

        adjustment_time:
            The time over which a party closes the gap to its desired stock.

        perception_delay:
            The time over which a party perceives the stock of the other.

    Returns:
        system:
            The model, with the stocks `race.a` and `race.b` in years, and the flows `race.build_a` and
            `race.build_b`. The observables of the model are `race.desired_a` and `race.desired_b`.

    Examples:
        >>> from skcomplex.models.social.organizations import escalation
        >>> escalation(margin=0.2).elements[0].margin
        0.2
    """
    race = ArmsRaceElement(
        'race',
        a=a,
        b=b,
        margin=margin,
        adjustment_time=adjustment_time,
        perception_delay=perception_delay,
    )
    return System(elements=[race])
