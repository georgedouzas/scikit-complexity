"""Stochastic process models: counting processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, JumpLaw, RuleLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView


class PoissonCounterElement(Element):
    """Poisson counter element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[object, StateSpec(domain=NN)] = 0.0,
        rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.rate = rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the poisson counter element."""
        return [JumpLaw(s.rate, {s.count: 1})]


class BernoulliTrialElement(Element):
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        agent.outcome = float(model.rng.random() < agent.p)
        agent.successes += agent.outcome


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
    return System(elements=[GroupElement('trials', element=BernoulliTrialElement, size=n_sequences, values={'p': p})])


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
    return System(
        elements=[GroupElement('counters', element=PoissonCounterElement, size=n_processes, values={'rate': rate})],
    )
