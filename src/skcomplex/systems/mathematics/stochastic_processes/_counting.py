"""Stochastic process systems of counting processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import JumpInteraction, MapInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PoissonCounterElement(Element):
    """Counter of the events of a Poisson process.

    The counter has the rate of events it is counted at.

    Args:
        label:
            The label of the counter or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        count:
            The initial number of events.

        rate:
            The rate of events, which the jump of the counter reads.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PoissonCounterElement
        >>> PoissonCounterElement('counter', rate=2.0).rate
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[float, StateSpec(domain=NN)] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.rate = rate


class BernoulliTrialView(AgentView):
    """View of a Bernoulli trial, with its states and parameters as attributes."""

    outcome: float
    successes: float
    p: float


class BernoulliTrialElement(Element):
    """Sequence of independent trials and the count of its successes.

    Each trial is a success with the same probability.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        outcome:
            The initial outcome, 1 for a success and 0 for a failure.

        successes:
            The initial number of successes.

        p:
            The probability of a success.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import BernoulliTrialElement
        >>> group = GroupElement('trials', element=BernoulliTrialElement, size=4, values={'p': 0.2})
        >>> group.label, group.element.__name__, group.size
        ('trials', 'BernoulliTrialElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('histograms', {'color_by': 'successes'}),
        *,
        outcome: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        successes: Annotated[float, StateSpec(domain=NN)] = 0.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.outcome = outcome
        self.successes = successes
        self.p = p

    @classmethod
    def _get_view(cls) -> type[BernoulliTrialView]:
        """Return the view of the agents.

        The view types their states and parameters.
        """
        return BernoulliTrialView


class BernoulliProcessSystem(BaseSystem):
    """Independent Bernoulli processes, one success or failure per step.

    Their success counts are binomial.

    The system has the group `trials` of states `outcome` and `successes`.

    Args:
        n_sequences:
            The number of independent sequences of trials.

        p:
            The probability of a success.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BernoulliProcessSystem
        >>> BernoulliProcessSystem(3).prepare().system_.elements[0].prepare().size_
        3
    """

    draws = True

    field = 'mathematics/stochastic_processes'
    topics = ('counting process', 'agents', 'independent trials')
    references = ('Thurner et al. 2018',)
    until = 100

    def __init__(
        self,
        n_sequences: int = 20,
        *,
        p: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sequences = n_sequences
        self.p = p

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        trial = MapInteraction(
            'trial',
            target='trials',
            state='outcome',
            function=lambda agent, system: float(system.rng.random() < agent.p),
            form='agent',
        )
        count = MapInteraction(
            'count',
            target='trials',
            state='successes',
            function=lambda agent, system: agent.outcome,
            mode='increment',
            form='agent',
        )
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement('trials', element=BernoulliTrialElement, size=self.n_sequences, values={'p': self.p}),
            ],
            interactions=[trial, count],
        )


class PoissonProcessSystem(BaseSystem):
    """Independent Poisson processes.

    The counters grow by one at exponentially distributed waiting times.

    The system has the counts `counters.count`.

    Args:
        n_processes:
            The number of independent counters.

        rate:
            The rate of events of a counter.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem
        >>> PoissonProcessSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'mathematics/stochastic_processes'
    topics = ('counting process', 'jump process', 'waiting times')
    references = ('Thurner et al. 2018',)
    until = 50.0
    dt = 1.0

    def __init__(
        self,
        n_processes: int = 20,
        *,
        rate: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_processes = n_processes
        self.rate = rate

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        counters = GroupElement(
            'counters',
            element=PoissonCounterElement,
            size=self.n_processes,
            values={'rate': self.rate},
        )
        events = JumpInteraction('events', target='counters', changes={'count': 1}, rate=lambda s: s.target.rate)
        return CompositeSystem(self.label, elements=[counters], interactions=[events])
