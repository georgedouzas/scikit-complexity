"""Stochastic process systems of counting processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role


class PoissonCounterElement(Element):
    """Counters of the events of Poisson processes, one count per process.

    Args:
        label:
            The label of the counters.

        count:
            The initial number of events, one per counter or shared.

        size:
            The number of counters, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PoissonCounterElement
        >>> PoissonCounterElement('counter', count=2.0).count
        2.0
    """

    def __init__(self, label: str, *, count: Annotated[float, NN] = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.count = count


class PoissonEventsInteraction(Interaction):
    """The events of a Poisson process, which add one to a count at the rate `rate`.

    Args:
        label:
            The label of the interaction.

        target:
            The counters whose counts the events add to.

        rate:
            The rate of the events.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PoissonEventsInteraction
        >>> PoissonEventsInteraction('events', target='counters', rate=2.0).rate
        2.0
    """

    law_classes = (JumpLaw,)

    def __init__(self, label: str, *, target: Role = 'counters', rate: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event that adds one to the count as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [JumpLaw(context.rate, {context.target.count: 1})]


class BernoulliTrialView(AgentViewProtocol, Protocol):
    """View of a sequence of Bernoulli trials, with its values as attributes."""

    outcome: float
    successes: float


class BernoulliTrialElement(Element):
    """Sequences of trials, each with the outcome `outcome` of its last trial and its number `successes` of successes.

    Args:
        label:
            The label of the sequences.

        outcome:
            The initial outcome, 1 for a success and 0 for a failure, one per sequence or shared.

        successes:
            The initial number of successes, one per sequence or shared.

        size:
            The number of sequences, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BernoulliTrialElement
        >>> BernoulliTrialElement('trials', successes=2.0, size=4).successes
        2.0
    """

    visual = VisualSpec('histograms', {'color_by': 'successes'})

    def __init__(
        self,
        label: str,
        *,
        outcome: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        successes: Annotated[object, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.outcome = outcome
        self.successes = successes


class BernoulliTrialInteraction(Interaction):
    """A Bernoulli trial at every step, a success with the probability `p`, counted in `successes`.

    Every sequence runs the rule once per step.

    Args:
        label:
            The label of the interaction.

        target:
            The sequences whose outcomes and numbers of successes the rule sets.

        p:
            The probability of a success.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BernoulliTrialInteraction
        >>> BernoulliTrialInteraction('trial', p=0.2).p
        0.2
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'trials', p: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.p = p

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the trial of every sequence as the rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self._run_trial)]

    def _run_trial(self, agent: BernoulliTrialView, system: SystemView) -> None:
        """Run one trial of a sequence and count its success."""
        agent.outcome = float(system.rng.random() < self.p)
        agent.successes += agent.outcome


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BernoulliProcessSystem
        >>> BernoulliProcessSystem(3).prepare().system_.elements[0].size
        3
    """

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sequences = n_sequences
        self.p = p

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the sequences and their trials."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[BernoulliTrialElement('trials', size=self.n_sequences)],
            interactions=[BernoulliTrialInteraction('trial', target='trials', p=self.p)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem
        >>> PoissonProcessSystem(5).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_processes = n_processes
        self.rate = rate

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the counters and their events."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[PoissonCounterElement('counters', size=self.n_processes)],
            interactions=[PoissonEventsInteraction('events', target='counters', rate=self.rate)],
        )
