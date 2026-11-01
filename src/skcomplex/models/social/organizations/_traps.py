"""Organization models: Meadows' system traps of success to the successful, shifting the burden, and escalation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import TransferInteraction
from ....core.laws import Sink, Source
from ....core.laws.utils import distributed_delay, lookup, where
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

ALLOCATION_TABLE = ((0.0, 0.0), (0.2, 0.05), (0.5, 0.5), (0.8, 0.95), (1.0, 1.0))


def _success(s: SimpleNamespace) -> Expression:
    """Return the past success of the first competitor, its fraction of all resources."""
    return s.a / (s.a + s.b)


def _allocation(s: SimpleNamespace) -> Expression:
    """Return the share of the budget the first competitor gets."""
    return lookup(_success(s), ALLOCATION_TABLE)


def _allocation_a(s: SimpleNamespace) -> Expression:
    """Return the share of the budget that flows to the first competitor."""
    return s.source.budget * _allocation(s.source)


def _allocation_b(s: SimpleNamespace) -> Expression:
    """Return the share of the budget that flows to the second competitor."""
    return s.source.budget * (1 - _allocation(s.source))


def _depreciation_a(s: SimpleNamespace) -> Expression:
    """Return the depreciation of the resources of the first competitor over their lifetime."""
    return s.source.a / s.source.lifetime


def _depreciation_b(s: SimpleNamespace) -> Expression:
    """Return the depreciation of the resources of the second competitor over their lifetime."""
    return s.source.b / s.source.lifetime


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


def _symptom(s: SimpleNamespace) -> Expression:
    """Return the symptom, the gap between the desired and the current state of the problem."""
    return s.desired - s.capability - s.intervention


def _fixing(s: SimpleNamespace) -> Expression:
    """Return the quick fix, in proportion to the symptom."""
    return s.source.fix_rate * _symptom(s.source)


def _building(s: SimpleNamespace) -> Expression:
    """Return the building of capability, the delayed symptom over the build time."""
    burden = s.source
    return distributed_delay(_symptom(burden), conserved=False, time=burden.solution_delay) / burden.build_time


def _erosion(s: SimpleNamespace) -> Expression:
    """Return the erosion of capability by the intervention."""
    return s.source.erosion * s.source.intervention * s.source.capability


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


def _desired_a(s: SimpleNamespace) -> Expression:
    """Return the stock the first party desires, a margin above its delayed view of the second's."""
    return (1 + s.margin) * distributed_delay(s.b, conserved=False, time=s.perception_delay)


def _desired_b(s: SimpleNamespace) -> Expression:
    """Return the stock the second party desires, a margin above its delayed view of the first's."""
    return (1 + s.margin) * distributed_delay(s.a, conserved=False, time=s.perception_delay)


def _build_a(s: SimpleNamespace) -> Expression:
    """Return the build-up of the first party toward its desired stock."""
    race = s.source
    desired = _desired_a(race)
    return where(desired > race.a, desired - race.a, 0) / race.adjustment_time


def _build_b(s: SimpleNamespace) -> Expression:
    """Return the build-up of the second party toward its desired stock."""
    race = s.source
    desired = _desired_b(race)
    return where(desired > race.b, desired - race.b, 0) / race.adjustment_time


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
    allocation = TransferInteraction(
        'allocation',
        source='contest',
        target='contest',
        state=Source('budget'),
        into='a',
        amount=_allocation_a,
        flow='allocation_a',
        legs=[
            (Source('budget'), 'b', _allocation_b, 'allocation_b'),
            ('a', Sink('depreciation'), _depreciation_a, 'depreciation_a'),
            ('b', Sink('depreciation'), _depreciation_b, 'depreciation_b'),
        ],
    )
    return System(
        elements=[ContestElement('contest', a=a, b=b, budget=budget, lifetime=lifetime)],
        interactions=[allocation],
    )


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
    flows = TransferInteraction(
        'flows',
        source='burden',
        target='burden',
        state=Source('intervener'),
        into='intervention',
        amount=_fixing,
        flow='fixing',
        legs=[
            (Source('learning'), 'capability', _building, 'building'),
            ('capability', Sink('atrophy'), _erosion, 'erosion'),
        ],
    )
    return System(elements=[burden], interactions=[flows])


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
    build = TransferInteraction(
        'build',
        source='race',
        target='race',
        state=Source('production'),
        into='a',
        amount=_build_a,
        flow='build_a',
        legs=[(Source('production'), 'b', _build_b, 'build_b')],
    )
    return System(elements=[race], interactions=[build])
