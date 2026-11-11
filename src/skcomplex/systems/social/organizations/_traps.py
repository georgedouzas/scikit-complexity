"""Organization systems: Meadows' system traps.

They run from success to the successful to drift to low performance.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.laws.utils import distributed_delay, series, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

ACTORS = ('a', 'b', 'c')
ALLOCATION_TABLE = ((0.0, 0.2, 0.5, 0.8, 1.0), (0.0, 0.05, 0.5, 0.95, 1.0))


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


class PerformanceElement(Element):
    """Performance element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        performance: Annotated[object, StateSpec(quantity='performance')] = 1.0,
        perceived: Annotated[object, StateSpec(quantity='performance')] = 1.0,
        desired: Annotated[object, StateSpec(quantity='performance')] = 1.0,
        bias: Annotated[object, ParameterSpec(domain=RealSet.open(-oo, 1))] = 0.1,
        perception_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        correction_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        goal_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        erosion: Annotated[object, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.performance = performance
        self.perceived = perceived
        self.desired = desired
        self.bias = bias
        self.perception_time = perception_time
        self.correction_time = correction_time
        self.goal_time = goal_time
        self.erosion = erosion


class StandoffElement(Element):
    """Standoff element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stock: Annotated[object, StateSpec(quantity='stock')] = 50.0,
        effort_a: Annotated[object, StateSpec(quantity='effort')] = 0.0,
        effort_b: Annotated[object, StateSpec(quantity='effort')] = 0.0,
        effort_c: Annotated[object, StateSpec(quantity='effort')] = 0.0,
        goal_a: Annotated[object, ParameterSpec()] = 80.0,
        goal_b: Annotated[object, ParameterSpec()] = 20.0,
        goal_c: Annotated[object, ParameterSpec()] = 50.0,
        strength_a: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        strength_b: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        strength_c: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        intensification_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stock = stock
        self.effort_a = effort_a
        self.effort_b = effort_b
        self.effort_c = effort_c
        self.goal_a = goal_a
        self.goal_b = goal_b
        self.goal_c = goal_c
        self.strength_a = strength_a
        self.strength_b = strength_b
        self.strength_c = strength_c
        self.intensification_time = intensification_time


def _get_success(s: SimpleNamespace) -> Expression:
    """Return the past success of the first competitor, its fraction of all resources."""
    return s.a / (s.a + s.b)


def _get_allocation(s: SimpleNamespace) -> Expression:
    """Return the share of the budget the first competitor gets."""
    return series(_get_success(s), *ALLOCATION_TABLE)


class SuccessToTheSuccessfulSystem(BaseSystem):
    """Meadows' success to the successful trap.

    Two competitors share a budget by their past success. Each competitor holds a stock of resources that its share of
    the budget fills and that depreciates over its lifetime. The share of the first competitor is a table function of
    its success, its fraction of all resources, that gives more than a proportional share to the more successful one.
    The two reinforcing loops make the symmetric state unstable, so the competitor that starts larger ends with the
    whole budget and the other one disappears. The book gives the structure only, so the table and the parameters are a
    documented choice.

    The system has the resources `contest.a` and `contest.b` in years, the flows `contest.allocation_a`,
    `contest.allocation_b`, `contest.depreciation_a`, and `contest.depreciation_b`. The observables of the system are
    `contest.success` and `contest.allocation`.

    Args:
        a:
            The initial resources of the first competitor.

        b:
            The initial resources of the second competitor.

        budget:
            The resources allocated per year.

        lifetime:
            The average lifetime of resources, in years.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import SuccessToTheSuccessfulSystem
        >>> SuccessToTheSuccessfulSystem(a=60.0).prepare().system_.elements[0].a
        60.0
    """

    field = 'social/organizations'
    topics = ('system traps', 'competitive exclusion', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 0.5
    observables = (
        ObservableSpec('success', _get_success, element='contest'),
        ObservableSpec('allocation', _get_allocation, element='contest'),
    )

    def __init__(
        self,
        a: float = 55.0,
        b: float = 45.0,
        *,
        budget: float = 10.0,
        lifetime: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.budget = budget
        self.lifetime = lifetime

    @staticmethod
    def _get_allocation_a(s: SimpleNamespace) -> Expression:
        """Return the share of the budget that flows to the first competitor."""
        return s.source.budget * _get_allocation(s.source)

    @staticmethod
    def _get_allocation_b(s: SimpleNamespace) -> Expression:
        """Return the share of the budget that flows to the second competitor."""
        return s.source.budget * (1 - _get_allocation(s.source))

    @staticmethod
    def _get_depreciation_a(s: SimpleNamespace) -> Expression:
        """Return the depreciation of the resources of the first competitor.

        The resources depreciate over their lifetime.
        """
        return s.source.a / s.source.lifetime

    @staticmethod
    def _get_depreciation_b(s: SimpleNamespace) -> Expression:
        """Return the depreciation of the resources of the second competitor.

        The resources depreciate over their lifetime.
        """
        return s.source.b / s.source.lifetime

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        allocation = TransferInteraction(
            'allocation',
            source='contest',
            target='contest',
            state=Source('budget'),
            into='a',
            amount=self._get_allocation_a,
            flow='allocation_a',
            legs=[
                (Source('budget'), 'b', self._get_allocation_b, 'allocation_b'),
                ('a', Sink('depreciation'), self._get_depreciation_a, 'depreciation_a'),
                ('b', Sink('depreciation'), self._get_depreciation_b, 'depreciation_b'),
            ],
        )
        return CompositeSystem(
            self.label,
            elements=[ContestElement('contest', a=self.a, b=self.b, budget=self.budget, lifetime=self.lifetime)],
            interactions=[allocation],
        )


def _get_symptom(s: SimpleNamespace) -> Expression:
    """Return the symptom.

    It is the gap between the desired and the current state of the problem.
    """
    return s.desired - s.capability - s.intervention


class ShiftingTheBurdenSystem(BaseSystem):
    """Meadows' shifting the burden trap.

    A quick fix relieves a symptom and erodes the system's capability. The symptom is the gap between the desired state
    and the state that the capability of the system and an outside intervention deliver together. The intervention grows
    quickly with the symptom. The capability grows slowly with the symptom, perceived over the solution delay, and the
    intervention erodes it as a side effect. With a quick fix the intervention takes over and the capability withers
    away, an addiction. Without one, `fix_rate = 0`, the capability grows to the desired state. The book gives the
    structure only, so the parameters are a documented choice.

    The system has the capability `burden.capability` and the intervention `burden.intervention` in years, the flows
    `burden.fixing`, `burden.building`, and `burden.erosion`. The observable of the system is `burden.symptom`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import ShiftingTheBurdenSystem
        >>> ShiftingTheBurdenSystem(fix_rate=0.0).prepare().system_.elements[0].fix_rate
        0.0
    """

    field = 'social/organizations'
    topics = ('system traps', 'addiction', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 60.0
    dt = 0.5
    observables = (ObservableSpec('symptom', _get_symptom, element='burden'),)

    def __init__(
        self,
        capability: float = 0.5,
        intervention: float = 0.0,
        *,
        desired: float = 1.0,
        fix_rate: float = 1.0,
        build_time: float = 5.0,
        solution_delay: float = 2.0,
        erosion: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.capability = capability
        self.intervention = intervention
        self.desired = desired
        self.fix_rate = fix_rate
        self.build_time = build_time
        self.solution_delay = solution_delay
        self.erosion = erosion

    @staticmethod
    def _get_fixing(s: SimpleNamespace) -> Expression:
        """Return the quick fix, in proportion to the symptom."""
        return s.source.fix_rate * _get_symptom(s.source)

    @staticmethod
    def _get_building(s: SimpleNamespace) -> Expression:
        """Return the building of capability, the delayed symptom over the build time."""
        burden = s.source
        return distributed_delay(_get_symptom(burden), conserved=False, time=burden.solution_delay) / burden.build_time

    @staticmethod
    def _get_erosion(s: SimpleNamespace) -> Expression:
        """Return the erosion of capability by the intervention."""
        return s.source.erosion * s.source.intervention * s.source.capability

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        burden = BurdenElement(
            'burden',
            capability=self.capability,
            intervention=self.intervention,
            desired=self.desired,
            fix_rate=self.fix_rate,
            build_time=self.build_time,
            solution_delay=self.solution_delay,
            erosion=self.erosion,
        )
        flows = TransferInteraction(
            'flows',
            source='burden',
            target='burden',
            state=Source('intervener'),
            into='intervention',
            amount=self._get_fixing,
            flow='fixing',
            legs=[
                (Source('learning'), 'capability', self._get_building, 'building'),
                ('capability', Sink('atrophy'), self._get_erosion, 'erosion'),
            ],
        )
        return CompositeSystem(self.label, elements=[burden], interactions=[flows])


def _get_desired_a(s: SimpleNamespace) -> Expression:
    """Return the stock the first party desires.

    It is a margin above its delayed view of the second's.
    """
    return (1 + s.margin) * distributed_delay(s.b, conserved=False, time=s.perception_delay)


def _get_desired_b(s: SimpleNamespace) -> Expression:
    """Return the stock the second party desires.

    It is a margin above its delayed view of the first's.
    """
    return (1 + s.margin) * distributed_delay(s.a, conserved=False, time=s.perception_delay)


class EscalationSystem(BaseSystem):
    """Meadows' escalation trap.

    Two parties each want to exceed the other's stock by a margin. Each party sees the stock of the other through an
    information smoothing over the perception delay, wants to hold `1 + margin` times that stock, and builds up the gap
    over the adjustment time. The loop through both stocks is reinforcing, so both stocks grow exponentially at the rate
    `r` that solves `r * adjustment_time = (1 + margin) / (1 + r * perception_delay) - 1`. The book gives the structure
    only, so the parameters are a documented choice.

    The system has the stocks `race.a` and `race.b` in years, and the flows `race.build_a` and `race.build_b`. The
    observables of the system are `race.desired_a` and `race.desired_b`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import EscalationSystem
        >>> EscalationSystem(margin=0.2).prepare().system_.elements[0].margin
        0.2
    """

    field = 'social/organizations'
    topics = ('system traps', 'arms race', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 0.5
    observables = (
        ObservableSpec('desired_a', _get_desired_a, element='race'),
        ObservableSpec('desired_b', _get_desired_b, element='race'),
    )

    def __init__(
        self,
        a: float = 10.0,
        b: float = 10.5,
        *,
        margin: float = 0.1,
        adjustment_time: float = 2.0,
        perception_delay: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.margin = margin
        self.adjustment_time = adjustment_time
        self.perception_delay = perception_delay

    @staticmethod
    def _build_a(s: SimpleNamespace) -> Expression:
        """Return the build-up of the first party toward its desired stock."""
        race = s.source
        desired = _get_desired_a(race)
        return where(desired > race.a, desired - race.a, 0) / race.adjustment_time

    @staticmethod
    def _build_b(s: SimpleNamespace) -> Expression:
        """Return the build-up of the second party toward its desired stock."""
        race = s.source
        desired = _get_desired_b(race)
        return where(desired > race.b, desired - race.b, 0) / race.adjustment_time

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        race = ArmsRaceElement(
            'race',
            a=self.a,
            b=self.b,
            margin=self.margin,
            adjustment_time=self.adjustment_time,
            perception_delay=self.perception_delay,
        )
        build = TransferInteraction(
            'build',
            source='race',
            target='race',
            state=Source('production'),
            into='a',
            amount=self._build_a,
            flow='build_a',
            legs=[(Source('production'), 'b', self._build_b, 'build_b')],
        )
        return CompositeSystem(self.label, elements=[race], interactions=[build])


def _get_compromise(s: SimpleNamespace) -> Expression:
    """Return the goal the stock settles at.

    It is the mean of the goals weighted by the strengths of the actors.
    """
    strengths = [getattr(s, f'strength_{actor}') for actor in ACTORS]
    goals = [getattr(s, f'goal_{actor}') for actor in ACTORS]
    return sum(strength * goal for strength, goal in zip(strengths, goals, strict=True)) / sum(strengths)


class PolicyResistanceSystem(BaseSystem):
    """Meadows' policy resistance trap.

    Three actors pull one stock toward inconsistent goals. Every actor compares the stock with its own goal and pulls
    the stock toward it, in proportion to the discrepancy and to its strength, and intensifies its effort while the
    discrepancy persists, the ratchet of the book. The book's example is the drug supply on the streets, which addicts
    want high, enforcement agencies want low, and pushers want in the middle. The countermoves produce a standoff: the
    stock settles at the mean of the goals weighted by the strengths, which none of the actors wants, while the efforts
    of the actors above and below it grow without bound. When one actor lets up, its strength set to zero, the others
    drag the stock toward their own goals. The book gives the structure only, as a narrative, so the corrective rules
    and the parameters are a documented choice.

    The system has the stock `standoff.stock` and the efforts `standoff.effort_a`, `standoff.effort_b`, and
    `standoff.effort_c`, and the flows `standoff.push`, `standoff.intensify_a`, `standoff.intensify_b`, and
    `standoff.intensify_c`. The observable of the system is `standoff.compromise`.

    Args:
        stock:
            The initial stock.

        goals:
            The goals of the three actors.

        strengths:
            The rates at which the three actors pull the stock toward their goals, per unit of discrepancy.

        intensification_time:
            The time over which an actor doubles its effort against a persistent discrepancy.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import PolicyResistanceSystem
        >>> PolicyResistanceSystem(strengths=(0.1, 0.0, 0.1)).prepare().system_.elements[0].strength_b
        0.0
    """

    field = 'social/organizations'
    topics = ('system traps', 'policy resistance', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 0.5
    observables = (ObservableSpec('compromise', _get_compromise, element='standoff'),)

    @staticmethod
    def _pull(s: SimpleNamespace, actor: str) -> Expression:
        """Return the corrective pull of an actor.

        It is its strength times the gap from its goal to the stock.
        """
        return getattr(s, f'strength_{actor}') * (getattr(s, f'goal_{actor}') - s.stock)

    def __init__(
        self,
        stock: float = 50.0,
        *,
        goals: tuple[float, float, float] = (80.0, 20.0, 50.0),
        strengths: tuple[float, float, float] = (0.1, 0.1, 0.1),
        intensification_time: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.stock = stock
        self.goals = goals
        self.strengths = strengths
        self.intensification_time = intensification_time

    @staticmethod
    def _push(s: SimpleNamespace) -> Expression:
        """Return the change of the stock.

        It is the corrective pulls and the efforts of all three actors.
        """
        standoff = s.source
        return sum(
            PolicyResistanceSystem._pull(standoff, actor) + getattr(standoff, f'effort_{actor}') for actor in ACTORS
        )

    @staticmethod
    def _intensify_a(s: SimpleNamespace) -> Expression:
        """Return the intensification of the effort of the first actor.

        The effort intensifies while the stock misses its goal.
        """
        return PolicyResistanceSystem._pull(s.source, 'a') / s.source.intensification_time

    @staticmethod
    def _intensify_b(s: SimpleNamespace) -> Expression:
        """Return the intensification of the effort of the second actor.

        The effort intensifies while the stock misses its goal.
        """
        return PolicyResistanceSystem._pull(s.source, 'b') / s.source.intensification_time

    @staticmethod
    def _intensify_c(s: SimpleNamespace) -> Expression:
        """Return the intensification of the effort of the third actor.

        The effort intensifies while the stock misses its goal.
        """
        return PolicyResistanceSystem._pull(s.source, 'c') / s.source.intensification_time

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        standoff = StandoffElement(
            'standoff',
            stock=self.stock,
            goal_a=self.goals[0],
            goal_b=self.goals[1],
            goal_c=self.goals[2],
            strength_a=self.strengths[0],
            strength_b=self.strengths[1],
            strength_c=self.strengths[2],
            intensification_time=self.intensification_time,
        )
        actions = TransferInteraction(
            'actions',
            source='standoff',
            target='standoff',
            state=Source('actors'),
            into='stock',
            amount=self._push,
            flow='push',
            legs=[
                (Source('actors'), 'effort_a', self._intensify_a, 'intensify_a'),
                (Source('actors'), 'effort_b', self._intensify_b, 'intensify_b'),
                (Source('actors'), 'effort_c', self._intensify_c, 'intensify_c'),
            ],
        )
        return CompositeSystem(self.label, elements=[standoff], interactions=[actions])


def _get_discrepancy(s: SimpleNamespace) -> Expression:
    """Return the discrepancy between the desired state and the perceived state."""
    return s.desired - s.perceived


class DriftToLowPerformanceSystem(BaseSystem):
    """Meadows' drift to low performance trap.

    The goals erode, following a pessimistic view of the performance. The actor compares the desired state with the
    actual performance and corrects the gap over the correction time, an ordinary balancing loop. The actor perceives
    the performance worse than it is, smoothed over the perception time and lowered by the negative bias, and the
    desired state slips toward the perceived state over the goal time. The lower the perceived state, the lower the
    goal, the less corrective action, and the lower the performance, a reinforcing loop that drags the performance down
    exponentially. With absolute standards, `absolute=True`, the desired state does not slip and the performance holds,
    and with an upbeat bias, a negative `bias`, the same structure drifts toward high performance, the two ways out of
    the book. The book gives the structure only, as a narrative, so the parameters are a documented choice.

    The system has the performance `performance.performance`, the perceived state `performance.perceived`, and the
    desired state `performance.desired`, and the flows `performance.correction`, `performance.perception`, and
    `performance.goal_erosion`. The observable of the system is `performance.discrepancy`.

    Args:
        performance:
            The initial performance, also the initial perceived and desired states.

        bias:
            The fraction by which the actor underrates the performance, negative for an upbeat bias.

        perception_time:
            The time over which the actor perceives the performance.

        correction_time:
            The time over which the corrective action closes the gap to the desired state.

        goal_time:
            The time over which the desired state follows the perceived state.

        absolute:
            Whether the desired state is kept absolute, regardless of the performance.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import DriftToLowPerformanceSystem
        >>> DriftToLowPerformanceSystem(absolute=True).prepare().system_.elements[0].erosion
        0.0
    """

    field = 'social/organizations'
    topics = ('system traps', 'eroding goals', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 0.5
    observables = (ObservableSpec('discrepancy', _get_discrepancy, element='performance'),)

    def __init__(
        self,
        performance: float = 1.0,
        *,
        bias: float = 0.1,
        perception_time: float = 2.0,
        correction_time: float = 2.0,
        goal_time: float = 5.0,
        absolute: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.performance = performance
        self.bias = bias
        self.perception_time = perception_time
        self.correction_time = correction_time
        self.goal_time = goal_time
        self.absolute = absolute

    @staticmethod
    def _get_correction(s: SimpleNamespace) -> Expression:
        """Return the corrective action.

        It is the gap from the performance to the desired state over the correction time.
        """
        return (s.source.desired - s.source.performance) / s.source.correction_time

    @staticmethod
    def _get_perception(s: SimpleNamespace) -> Expression:
        """Return the update of the perceived state toward the biased performance.

        The performance is lowered by the negative bias.
        """
        performance = s.source
        return ((1 - performance.bias) * performance.performance - performance.perceived) / performance.perception_time

    @staticmethod
    def _get_goal_erosion(s: SimpleNamespace) -> Expression:
        """Return the slip of the desired state toward the perceived state.

        It is zero for absolute standards.
        """
        performance = s.source
        return performance.erosion * (performance.perceived - performance.desired) / performance.goal_time

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        element = PerformanceElement(
            'performance',
            performance=self.performance,
            perceived=self.performance,
            desired=self.performance,
            bias=self.bias,
            perception_time=self.perception_time,
            correction_time=self.correction_time,
            goal_time=self.goal_time,
            erosion=0.0 if self.absolute else 1.0,
        )
        flows = TransferInteraction(
            'flows',
            source='performance',
            target='performance',
            state=Source('effort'),
            into='performance',
            amount=self._get_correction,
            flow='correction',
            legs=[
                (Source('information'), 'perceived', self._get_perception, 'perception'),
                (Source('standards'), 'desired', self._get_goal_erosion, 'goal_erosion'),
            ],
        )
        return CompositeSystem(self.label, elements=[element], interactions=[flows])
