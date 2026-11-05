"""Stochastic process models: counting processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import GroupElement
from ....core.interactions import JumpInteraction, MapInteraction
from ....core.system import System
from ..._catalog import register_model
from ._bernoulli_trial import BernoulliTrialElement
from ._poisson_counter import PoissonCounterElement


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
        >>> bernoulli_process(3).elements[0].prepare().size_
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
        >>> poisson_process(5).elements[0].prepare().size_
        5
    """
    counters = GroupElement('counters', element=PoissonCounterElement, size=n_processes, values={'rate': rate})
    events = JumpInteraction('events', target='counters', changes={'count': 1}, rate=lambda s: s.target.rate)
    return System(elements=[counters], interactions=[events])
