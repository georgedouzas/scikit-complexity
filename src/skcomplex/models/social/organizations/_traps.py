"""Organization models: Meadows' system traps of success to the successful, shifting the burden, and escalation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Auxiliary, Element, Flow, Law, Sink, Source, Symbols, System, lookup, smooth, where
from ....core.runtime import register_model

ALLOCATION_TABLE = ((0.0, 0.0), (0.2, 0.05), (0.5, 0.5), (0.8, 0.95), (1.0, 1.0))


def _contest_laws(s: Symbols) -> list[Law]:
    """Return the laws of two competitors that share a budget in proportion to their past success."""
    success = Auxiliary('success', s.a / (s.a + s.b))
    allocation = Auxiliary('allocation', lookup(success.symbol, ALLOCATION_TABLE))
    return [
        success,
        allocation,
        Flow(Source('budget'), s.a, s.budget * allocation.symbol, label='allocation_a'),
        Flow(Source('budget'), s.b, s.budget * (1 - allocation.symbol), label='allocation_b'),
        Flow(s.a, Sink('depreciation'), s.a / s.lifetime, label='depreciation_a'),
        Flow(s.b, Sink('depreciation'), s.b / s.lifetime, label='depreciation_b'),
    ]


Contest = Element.define(
    'Contest',
    states={
        'a': state(default=55.0, domain=Domain.NONNEGATIVE, quantity='resources'),
        'b': state(default=45.0, domain=Domain.NONNEGATIVE, quantity='resources'),
    },
    parameters={
        'budget': parameter(default=10.0, domain=Domain.NONNEGATIVE),
        'lifetime': parameter(default=10.0, domain=Domain.POSITIVE),
    },
    laws=_contest_laws,
)


def _burden_laws(s: Symbols) -> list[Law]:
    """Return the laws of a problem that a quick fix relieves while it erodes the capability of a fundamental one."""
    symptom = Auxiliary('symptom', s.desired - s.capability - s.intervention)
    building = smooth(symptom.symbol, s.solution_delay) / s.build_time
    return [
        symptom,
        Flow(Source('intervener'), s.intervention, s.fix_rate * symptom.symbol, label='fixing'),
        Flow(Source('learning'), s.capability, building, label='building'),
        Flow(s.capability, Sink('atrophy'), s.erosion * s.intervention * s.capability, label='erosion'),
    ]


Burden = Element.define(
    'Burden',
    states={
        'capability': state(default=0.5, domain=Domain.NONNEGATIVE, quantity='capability'),
        'intervention': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='intervention'),
    },
    parameters={
        'desired': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'fix_rate': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'build_time': parameter(default=5.0, domain=Domain.POSITIVE),
        'solution_delay': parameter(default=2.0, domain=Domain.POSITIVE),
        'erosion': parameter(default=0.5, domain=Domain.NONNEGATIVE),
    },
    laws=_burden_laws,
)


def _escalation_laws(s: Symbols) -> list[Law]:
    """Return the laws of two parties that each build their stock up to a margin above their view of the other's."""
    desired_a = Auxiliary('desired_a', (1 + s.margin) * smooth(s.b, s.perception_delay))
    desired_b = Auxiliary('desired_b', (1 + s.margin) * smooth(s.a, s.perception_delay))
    build_a = where(desired_a.symbol > s.a, desired_a.symbol - s.a, 0) / s.adjustment_time
    build_b = where(desired_b.symbol > s.b, desired_b.symbol - s.b, 0) / s.adjustment_time
    return [
        desired_a,
        desired_b,
        Flow(Source('production'), s.a, build_a, label='build_a'),
        Flow(Source('production'), s.b, build_b, label='build_b'),
    ]


ArmsRace = Element.define(
    'ArmsRace',
    states={
        'a': state(default=10.0, domain=Domain.NONNEGATIVE, quantity='arms'),
        'b': state(default=10.5, domain=Domain.NONNEGATIVE, quantity='arms'),
    },
    parameters={
        'margin': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'adjustment_time': parameter(default=2.0, domain=Domain.POSITIVE),
        'perception_delay': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=_escalation_laws,
)


@register_model(
    field='social/organizations',
    tags=('system traps', 'competitive exclusion', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 0.5},
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
            `contest.allocation_b`, `contest.depreciation_a`, and `contest.depreciation_b`, and the auxiliaries
            `contest.success` and `contest.allocation`.

    Examples:
        >>> from skcomplex.models.social.organizations import success_to_the_successful
        >>> success_to_the_successful(a=60.0).elements[0].a
        60.0
    """
    return System(elements=[Contest('contest', a=a, b=b, budget=budget, lifetime=lifetime)])


@register_model(
    field='social/organizations',
    tags=('system traps', 'addiction', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 60.0, 'dt': 0.5},
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
            the flows `burden.fixing`, `burden.building`, and `burden.erosion`, and the auxiliary `burden.symptom`.

    Examples:
        >>> from skcomplex.models.social.organizations import shifting_the_burden
        >>> shifting_the_burden(fix_rate=0.0).elements[0].fix_rate
        0.0
    """
    burden = Burden(
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
            The model, with the stocks `race.a` and `race.b` in years, the flows `race.build_a` and `race.build_b`,
            and the auxiliaries `race.desired_a` and `race.desired_b`.

    Examples:
        >>> from skcomplex.models.social.organizations import escalation
        >>> escalation(margin=0.2).elements[0].margin
        0.2
    """
    race = ArmsRace(
        'race',
        a=a,
        b=b,
        margin=margin,
        adjustment_time=adjustment_time,
        perception_delay=perception_delay,
    )
    return System(elements=[race])
