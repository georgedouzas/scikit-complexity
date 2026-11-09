"""Define payoff learning.

Payoff learning moves the scores of options towards the payoffs they realized.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.sets.real_set import RealSet

from .....core.components.base import BasePart
from .....core.components.interactions import MapInteraction
from .....core.components.laws import UpdateLaw
from .....core.components.laws.utils import where
from .....core.components.spaces import BaseSpace
from .....core.components.types import AgentViewProtocol, SystemViewProtocol
from .....core.errors import ValidationError
from .....core.specs import ParameterSpec, RoleSpec
from .....core.types import Array, ElementKind

RULES = ('average', 'cumulative')


class PayoffLearningInteraction(MapInteraction):
    """Payoff learning of the scores of options.

    Economics names it reinforcement learning of payoffs, as in fictitious play and the attractions of
    experience-weighted learning. Each score moves towards the payoff its option realized in the step. Every step, the
    score `s_k` of option `k` of every member moves towards the realized payoff `P_k` at the rate `a`, `s_k <- s_k + a
    (P_k - s_k)`, so a score is an exponentially weighted mean of past payoffs, and the last payoff at the rate 1, by
    the rule `'average'`. By the rule `'cumulative'` the score adds the payoff, `s_k <- s_k + P_k`, so it is the sum of
    past payoffs, such as a virtual score. In the agent form the scores may be the components of one vector state, one
    per option, and a group of one member holds scores that several groups share. In the `'symbolic'` form,
    `payoff(namespace, k)` returns the payoff of option `k` as a symbolic expression, where `namespace.target` holds the
    states and parameters of the group. In the `'agent'` form, the group is an agent group and `payoff(agent, system,
    k)` returns the payoff as a number, so it can read the agent and the history and storage of the system view. With
    `chosen`, the name of a state that holds the index of the option a member chose, only the score of that option
    moves, and the payoff of the other options is not read. In the agent form every agent reinforces at once under the
    schedule `'synchronous'`, one after the other in their order or in a new random order every step under
    `'sequential'` or `'random-sequential'`, and each in its own activation under `'member'`, together with the rules of
    its element and the other member interactions, in the order the system lists them, such as after a choice and a
    transfer in one visit.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose members hold the scores.

        scores:
            The names of the states that hold the score of every option, in the order of the options, or in the agent
            form the name of one vector state whose components are the scores.

        payoff:
            The function `payoff(namespace, k)` in the symbolic form, or `payoff(agent, system, k)` in the agent form,
            that returns the realized payoff of option `k`.

        rate:
            The rate `a` at which the scores move towards the payoffs, between 0 and 1, by the rule `'average'`.

        rule:
            `'average'` to move the scores towards the payoffs, or `'cumulative'` to add the payoffs to the scores.

        form:
            `'symbolic'` for update laws of the states of a group, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

        chosen:
            The name of the state that holds the index of the option a member chose, so only its score moves, or
            `None` to move every score.

        schedule:
            The order of the agent form, `'synchronous'` for every agent at once, `'sequential'` or
            `'random-sequential'` for one agent after the other, or `'member'` for every agent in its own activation.

    Raises:
        ValidationError:
            The form, the rule, or the schedule is unknown, the symbolic form runs member by member, there are no
            scores, or the payoff is not callable.

    Examples:
        >>> from skcomplex.systems.economics.common.learning import PayoffLearningInteraction
        >>> reinforcement = PayoffLearningInteraction(
        ...     'reinforcement', target='learners', scores=('first', 'second'), payoff=lambda s, k: k, rate=0.5
        ... )
        >>> reinforcement.rate
        0.5
        >>> PayoffLearningInteraction(
        ...     'virtual', target='players', scores='scores', payoff=lambda a, m, k: k, rule='cumulative', form='agent'
        ... ).rule
        'cumulative'
        >>> PayoffLearningInteraction(
        ...     'satisfaction', target='customers', scores=('s1', 's2'), payoff=lambda a, m, k: 1.0, chosen='shop',
        ...     schedule='member', form='agent',
        ... ).chosen
        'shop'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('scores', 'chosen'))],
        scores: str | Sequence[str],
        payoff: Callable[..., float | Expression],
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        rule: str = 'average',
        form: str = 'symbolic',
        phase: str | None = None,
        chosen: str | None = None,
        schedule: str = 'synchronous',
    ) -> None:
        first = scores if isinstance(scores, str) else next(iter(scores), '')
        super().__init__(
            label,
            target=target,
            state=first,
            function=self._get_scores,
            form=form,
            phase=phase,
            schedule=schedule,
        )
        self.scores = scores
        self.payoff = payoff
        self.rate = rate
        self.rule = rule
        self.chosen = chosen

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown variant, no scores, or an uncallable payoff.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, the form, the rule, or the schedule is unknown, the symbolic form runs member
                by member, there are no scores, or the payoff is not callable.
        """
        super()._prepare(context)
        if self.rule not in RULES:
            error_msg = f'Unknown rule {self.rule!r} of {self.label!r}. Use one of {RULES}.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.schedule != 'synchronous':
            error_msg = f'The schedule {self.schedule!r} of {self.label!r} runs agent rules. Use `form="agent"`.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.schedule != 'synchronous':
            error_msg = f'The schedule {self.schedule!r} of {self.label!r} runs agent rules. Use `form="agent"`.'
            raise ValidationError(error_msg)
        vector = isinstance(self.scores, str) and bool(self.scores) and self.form == 'agent'
        if not vector and (isinstance(self.scores, str) or not self.scores):
            error_msg = (
                f'{self.label!r} has the scores {self.scores!r}. Give a sequence of state names, one per option, or '
                'in the agent form the name of a vector state.'
            )
            raise ValidationError(error_msg)
        if not callable(self.payoff):
            error_msg = f'The payoff of {self.label!r} must be callable, but it is {self.payoff!r}.'
            raise ValidationError(error_msg)

    def _read_scores(self, agent: AgentViewProtocol) -> Array:
        """Return the scores of an agent, one per option."""
        if isinstance(self.scores, str):
            return np.array(getattr(agent, self.scores), dtype=float).ravel()
        return np.array([getattr(agent, name) for name in self.scores], dtype=float)

    def _get_scores(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> Array:
        """Return the new scores of an agent.

        Every score, or only that of its chosen option, moves by its payoff.
        """
        old = self._read_scores(agent)
        options = range(len(old)) if self.chosen is None else [int(getattr(agent, self.chosen))]
        rate = 1.0 if self.rule == 'cumulative' else self._get_parameter_value('rate', system)
        change = np.zeros_like(old)
        for k in options:
            payoff = float(self.payoff(agent, system, k))
            change[k] = payoff if self.rule == 'cumulative' else rate * (payoff - old[k])
        return old + change

    def _get_value(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> Array:
        """Return the new scores of an agent, the next value of the map."""
        return self._get_scores(agent, system)

    def _set_value(self, agent: AgentViewProtocol, value: object) -> None:
        """Set the scores of an agent."""
        scores = np.asarray(value, dtype=float)
        if isinstance(self.scores, str):
            setattr(agent, self.scores, scores)
            return
        for name, score in zip(self.scores, scores, strict=True):
            setattr(agent, name, float(score))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of every score towards its payoff, or the agent-form rule.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.rate`.

        Returns:
            laws:
                The update laws of the scores, or the rule that reinforces the scores of the agents.
        """
        if self.form == 'agent':
            return super()._get_laws(namespace)
        laws: list[BasePart] = []
        for index, name in enumerate(self.scores):
            score = getattr(self._get_member(namespace, 'target'), name)
            payoff = self.payoff(namespace, index)
            change = (
                payoff
                if self.rule == 'cumulative'
                else self._get_parameter_symbol('rate', namespace) * (payoff - score)
            )
            if self.chosen is not None:
                change = where(getattr(self._get_member(namespace, 'target'), self.chosen) == index, change, 0)
            laws.append(UpdateLaw(score, score + change))
        return laws
