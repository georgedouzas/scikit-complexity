"""Stochastic process models: counting processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import JumpInteraction, MapInteraction
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._bernoulli_trial_view import BernoulliTrialView


class PoissonCounterElement(Element):
    """Counter of the events of a Poisson process, with the rate of events it is counted at.

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
        >>> from skcomplex.models.mathematics.stochastic_processes import PoissonCounterElement
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


class BernoulliTrialElement(Element):
    """Sequence of independent trials, each a success with the same probability, and the count of its successes.

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
        >>> from skcomplex.models.mathematics.stochastic_processes import BernoulliTrialElement
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
        """Return the view of the agents, which types their states and parameters."""
        return BernoulliTrialView


@register_model(
    field='mathematics/stochastic_processes',
    tags=('counting process', 'agents', 'independent trials'),
    references=('Thurner et al. 2018',),
    defaults={'until': 100},
)
def bernoulli_process(n_sequences: int = 20, *, p: float = 0.5) -> System:
    """Return independent Bernoulli processes, one success or failure per step, whose success counts are binomial.

    Args:
        n_sequences:
            The number of independent sequences of trials.

        p:
            The probability of a success.

    Returns:
        system:
            The model, with the group `trials` of states `outcome` and `successes`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import bernoulli_process
        >>> bernoulli_process(3).elements[0].check().prepare().size_
        3
    """
    trial = MapInteraction(
        'trial',
        target='trials',
        state='outcome',
        function=lambda agent, model: float(model.rng.random() < agent.p),
        form='agent',
    )
    count = MapInteraction(
        'count',
        target='trials',
        state='successes',
        function=lambda agent, model: agent.outcome,
        mode='increment',
        form='agent',
    )
    return System(
        elements=[GroupElement('trials', element=BernoulliTrialElement, size=n_sequences, values={'p': p})],
        interactions=[trial, count],
    )


@register_model(
    field='mathematics/stochastic_processes',
    tags=('counting process', 'jump process', 'waiting times'),
    references=('Thurner et al. 2018',),
    defaults={'until': 50.0, 'dt': 1.0},
)
def poisson_process(n_processes: int = 20, *, rate: float = 1.0) -> System:
    """Return independent Poisson processes, counters that grow by one at exponentially distributed waiting times.

    Args:
        n_processes:
            The number of independent counters.

        rate:
            The rate of events of a counter.

    Returns:
        system:
            The model, with the counts `counters.count`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import poisson_process
        >>> poisson_process(5).elements[0].check().prepare().size_
        5
    """
    counters = GroupElement('counters', element=PoissonCounterElement, size=n_processes, values={'rate': rate})
    events = JumpInteraction('events', target='counters', changes={'count': 1}, rate=lambda s: s.target.rate)
    return System(elements=[counters], interactions=[events])
