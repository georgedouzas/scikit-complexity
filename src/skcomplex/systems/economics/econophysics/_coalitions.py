"""Econophysics systems: El Farol patrons who form coalitions that share their best predictor of the attendance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import FeedbackInteraction
from ....core.components.laws import BaseLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class CoalitionPatronView(AgentView):
    """Patron as the bar sees it: its coalition, its attendance, and its return."""

    coalition: float
    attends: float
    score: float


class CoalitionPatronElement(Element):
    """Patron of the El Farol bar who belongs to a coalition and keeps a smoothed return of its nights.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        coalition:
            The label of the coalition of the patron.

        attends:
            1 when the patron went to the bar on the last night, 0 otherwise.

        score:
            The exponentially smoothed share of good nights, a night at the bar that was not overcrowded or a night
            away from an overcrowded bar.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.econophysics import CoalitionPatronElement
        >>> GroupElement('patrons', element=CoalitionPatronElement, size=3).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'coalition'}),
        *,
        coalition: Annotated[float, StateSpec(domain=NN)] = 0,
        attends: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        score: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.coalition = coalition
        self.attends = attends
        self.score = score

    @classmethod
    def _get_view(cls) -> type[CoalitionPatronView]:
        """Return the view of the agents of the element."""
        return CoalitionPatronView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the bar decides for every patron.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class CoalitionBarInteraction(FeedbackInteraction[SystemView]):
    """El Farol bar of Rand and Wilensky whose patrons share strategies within coalitions that split and merge.

    A strategy `s` in `[-1, 1]^(m + 1)` predicts the attendance `n s_1 + sum_j s_(j + 1) h_j` from the last `m`
    attendances `h`, and its error is the total absolute error of its predictions over the last `m` nights. Every night
    every coalition uses the strategy of the lowest error among the strategies of its members, and every member goes
    when the prediction is at most the capacity. Every member's return moves by `smoothing` towards 1 after a good
    night and towards 0 after a bad one. With coalitions, after every night (Collins 2019, Section 3.2), every
    coalition in random order that attended an overcrowded night within the memory, when the smallest excess `x` of
    those nights is below its size `N`, splits off a random subgroup of at most `N - x` members that holds the owner of
    its best strategy. Then, at most once per coalition and night, two coalitions that attended an uncrowded night
    merge unless their joint size would have overcrowded an uncrowded night either attended within the memory. A merged
    coalition keeps the longer history.

    Args:
        label:
            The label of the bar.

        rule:
            Unused, since the bar overrides `act`.

        phase:
            The phase the bar acts in, or `None` for the default phase.

        patrons:
            The path of the group of patrons.

        strategies:
            The strategies of every patron, `strategies[i][k]` the `m + 1` weights of strategy `k` of patron `i`.

        capacity:
            The attendance above which the bar is overcrowded.

        smoothing:
            The weight `lambda` of the last night in the return.

        coalitions:
            Whether patrons form coalitions.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import CoalitionBarInteraction
        >>> CoalitionBarInteraction('bar', patrons='patrons').reports[:2]
        ('attendance', 'coalitions')
    """

    draws = True

    reports: ClassVar[tuple[str, ...]] = ('attendance', 'coalitions', 'mean_score', 'splits', 'joins')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        patrons: Annotated[str, RoleSpec()],
        strategies: tuple[tuple[tuple[float, ...], ...], ...] = (),
        capacity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 60.0,
        smoothing: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        coalitions: bool = True,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.patrons = patrons
        self.strategies = strategies
        self.capacity = capacity
        self.smoothing = smoothing
        self.coalitions = coalitions

    @staticmethod
    def _errors(strategies: Array, history: Array) -> Array:
        """Return the total absolute error over the last `m` nights of every strategy, from the newest attendance."""
        memory = strategies.shape[-1] - 1
        size = len(strategies)
        features = np.array([[size, *history[night + 1 : night + 1 + memory]] for night in range(memory)])
        predictions = strategies @ features.T
        return np.abs(predictions - history[:memory]).sum(axis=-1)

    def _split(
        self,
        system: SystemView,
        labels: Array,
        best: dict[int, int],
        records: dict[int, list[tuple[float, float]]],
    ) -> int:
        """Split off subgroups of the coalitions that attended overcrowded nights, and return the number of splits."""
        rng, capacity, splits = system.rng, system.parameter('capacity'), 0
        for label in rng.permutation(np.unique(labels)):
            members = np.flatnonzero(labels == label)
            excesses = [attendance - capacity for attendance, _ in records.get(int(label), []) if attendance > capacity]
            if len(members) <= 1 or not excesses or min(excesses) >= len(members):
                continue
            largest = int(len(members) - min(excesses))
            size = int(rng.integers(1, min(largest, len(members) - 1) + 1))
            others = rng.permutation(members[members != best[int(label)]])[: size - 1]
            new = int(labels.max()) + 1
            labels[[best[int(label)], *others]] = new
            records[new] = list(records.get(int(label), []))
            splits += 1
        return splits

    def _join(
        self,
        system: SystemView,
        labels: Array,
        attended: Array,
        records: dict[int, list[tuple[float, float]]],
    ) -> int:
        """Merge pairs of coalitions that attended an uncrowded night without risking crowding, and return the joins."""
        capacity, joins = system.parameter('capacity'), 0
        present = [int(label) for label in system.rng.permutation(np.unique(labels[attended]))]
        merged: set[int] = set()
        for first in present:
            for second in present:
                if first >= second or first in merged or second in merged:
                    continue
                sizes = {label: int(np.sum(labels == label)) for label in (first, second)}
                safe = all(
                    attendance + sizes[second if label == first else first] <= capacity
                    for label in (first, second)
                    for attendance, _ in records.get(label, [])
                    if attendance <= capacity
                )
                if safe:
                    keep = max((first, second), key=lambda label: len(records.get(label, [])))
                    labels[labels == (second if keep == first else first)] = keep
                    merged.update((first, second))
                    joins += 1
        return joins

    def act(self, system: SystemView) -> None:
        """Decide the night, score the patrons, and let the coalitions split and merge.

        Args:
            system:
                The system view of the bar.
        """
        patrons = sorted(system.agents('patrons', CoalitionPatronView), key=lambda patron: patron.id)
        strategies = np.asarray(self.strategies, dtype=float)
        memory, size, capacity = strategies.shape[-1] - 1, len(patrons), system.parameter('capacity')
        stored = system.storage.get('history')
        history = np.asarray(stored) if isinstance(stored, np.ndarray) else system.rng.integers(0, size + 1, 2 * memory)
        records = system.storage.get('records')
        records = records if isinstance(records, dict) else {}
        labels = np.array([int(patron.coalition) for patron in patrons])
        if not self.coalitions:
            labels = np.arange(size)
        errors = self._errors(strategies, history.astype(float))
        best: dict[int, int] = {}
        attends = np.zeros(size, dtype=bool)
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            owner, strategy = np.unravel_index(np.argmin(errors[members]), errors[members].shape)
            best[int(label)] = int(members[owner])
            weights = strategies[members[owner], strategy]
            attends[members] = size * weights[0] + weights[1:] @ history[:memory] <= capacity
        attendance = float(attends.sum())
        crowded = attendance > capacity
        good = np.where(attends, not crowded, crowded)
        smoothing = system.parameter('smoothing')
        scores = np.array([patron.score for patron in patrons]) * (1 - smoothing) + smoothing * good
        for label in np.unique(labels[attends]):
            records.setdefault(int(label), []).append((attendance, float(np.sum(labels == label))))
        for label in list(records):
            records[label] = records[label][-memory:]
        splits = joins = 0
        if self.coalitions:
            splits = self._split(system, labels, best, records)
            joins = 0 if crowded else self._join(system, labels, attends, records)
        system.storage.update(history=np.concatenate(([attendance], history[:-1])), records=records)
        for patron, label, went, score in zip(patrons, labels, attends, scores, strict=True):
            patron.coalition, patron.attends, patron.score = float(label), float(went), float(score)
        system.report('attendance', attendance)
        system.report('coalitions', float(len(np.unique(labels))))
        system.report('mean_score', float(scores.mean()))
        system.report('splits', float(splits))
        system.report('joins', float(joins))


class ElFarolCoalitionsSystem(BaseSystem):
    """Collins' El Farol bar with strategic group formation, whose coalitions share their best strategy.

    Patrons of Rand and Wilensky's El Farol model predict the attendance with linear strategies of the last `memory`
    attendances and go when they predict room. In Collins' version (Carmichael et al. 2019, chapter 9) patrons form
    coalitions that use the best strategy of all their members, so a larger coalition has better predictions but adds
    more to the crowd. Coalitions split off the members with the best strategy after overcrowded nights, the core
    condition that no subgroup gains by leaving, and coalitions that attended together merge when it would not have
    crowded an earlier good night. Without coalitions the mean attendance stays near the capacity and patrons enjoy
    about half their nights, while coalitions make the attendance swing and lower the returns of all, because the
    patrons are boundedly rational. The return of a patron is exponentially smoothed with the weight `smoothing`,
    which the source leaves open.

    The system has the group `patrons` and the bar `bar`, whose reporters are `bar.attendance`, the number of coalitions
    `bar.coalitions`, the mean return `bar.mean_score`, and the splits `bar.splits` and joins `bar.joins` of the night.

    Args:
        n_agents:
            The number of patrons.

        capacity:
            The attendance above which the bar is overcrowded.

        memory:
            The number of past attendances a strategy reads.

        n_strategies:
            The number of strategies of every patron.

        smoothing:
            The weight of the last night in the return.

        coalitions:
            Whether patrons form coalitions.

        random_state:
            The seed or generator of the strategies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import ElFarolCoalitionsSystem
        >>> ElFarolCoalitionsSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'economics/econophysics'
    topics = ('el farol', 'agents', 'coalitions', 'group formation', 'cooperative game theory', 'bounded rationality')
    references = ('Carmichael et al. 2019 ch. 9',)
    until = 500

    def __init__(
        self,
        n_agents: int = 100,
        *,
        capacity: float = 60.0,
        memory: int = 5,
        n_strategies: int = 10,
        smoothing: float = 0.1,
        coalitions: bool = True,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.capacity = capacity
        self.memory = memory
        self.n_strategies = n_strategies
        self.smoothing = smoothing
        self.coalitions = coalitions
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        strategies = rng.uniform(-1.0, 1.0, (self.n_agents, self.n_strategies, self.memory + 1))
        patrons = GroupElement(
            'patrons',
            element=CoalitionPatronElement,
            size=self.n_agents,
            values={'coalition': np.arange(self.n_agents, dtype=float)},
        )
        bar = CoalitionBarInteraction(
            'bar',
            patrons='patrons',
            strategies=tuple(tuple(map(tuple, rows)) for rows in strategies.tolist()),
            capacity=self.capacity,
            smoothing=self.smoothing,
            coalitions=self.coalitions,
        )
        return System(self.label, elements=[patrons], interactions=[bar])
