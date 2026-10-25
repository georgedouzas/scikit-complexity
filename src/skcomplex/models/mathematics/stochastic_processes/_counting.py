"""Stochastic process models: counting processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.base import BaseLaw, NumericDomain, VisualSpec, parameter, state
from ....core.components import ElementComponent
from ....core.laws import JumpLaw, Symbols
from ....core.system import System
from ....engine.runtime import register_model
from ....formalisms.agents import AgentView, ModelView, RuleLaw

PoissonCounterComponent = ElementComponent.define(
    'PoissonCounterComponent',
    states={'count': state(default=0.0, domain=NumericDomain.NONNEGATIVE_INTEGER)},
    parameters={'rate': parameter(1.0, domain=NumericDomain.NONNEGATIVE)},
    laws=lambda s: [JumpLaw(s.rate, {s.count: 1})],
)


class BernoulliTrialComponent(ElementComponent):
    """Sequence of independent trials, each a success with the same probability.

    Args:
        label:
            The label of the agent or group.

        outcome:
            The initial outcome, 1 for a success and 0 for a failure.

        successes:
            The initial number of successes.

        p:
            The probability of a success.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import BernoulliTrialComponent
        >>> BernoulliTrialComponent.group('trials', 4, p=0.2)
        GroupComponent('trials', element=BernoulliTrialComponent, size=4)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('traits', {'color_by': 'successes'})
    outcome: float = state(default=0.0, domain=NumericDomain.BOOLEAN)
    successes: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE_INTEGER)
    p: float = parameter(0.5, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the trial rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.draw)]

    def draw(self, agent: AgentView, model: ModelView) -> None:
        """Draw the next outcome and count it.

        Args:
            agent:
                The sequence of trials.

            model:
                The model.
        """
        agent.set_state('outcome', float(model.rng.random() < agent.parameter('p')))
        agent.set_state('successes', agent.state('successes') + agent.state('outcome'))


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
        >>> bernoulli_process(3).elements[0].size
        3
    """
    return System(components=[BernoulliTrialComponent.group('trials', n_sequences, p=p)])


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
        >>> poisson_process(5).elements[0].size
        5
    """
    return System(components=[PoissonCounterComponent.group('counters', n_processes, rate=rate)])
