"""Define payoff learning.

Payoff learning moves the scores of options towards the payoffs they realized.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, UpdateLaw
from .....core.components.laws.utils import where
from .....core.components.spaces import BaseSpace
from .....core.components.types import AgentViewProtocol, SystemViewProtocol
from .....core.errors import ValidationError
from .....core.types import Array, Role

RULES = ('average', 'cumulative')
SCHEDULES = (None, 'synchronous', 'sequential', 'random-sequential', 'member')


class PayoffLearningInteraction(Interaction):
    """Payoff learning of the scores of options.

    Economics names it reinforcement learning of payoffs, as in fictitious play and the attractions of
    experience-weighted learning. Each score moves towards the payoff its option realized in the step. Every step, the
    score `s_k` of option `k` of every member moves towards the realized payoff `P_k` at the rate `a`, `s_k <- s_k + a
    (P_k - s_k)`, so a score is an exponentially weighted mean of past payoffs, and the last payoff at the rate 1, by
    the rule `'average'`. By the rule `'cumulative'` the score adds the payoff, `s_k <- s_k + P_k`, so it is the sum of
    past payoffs, such as a virtual score. Without a `schedule` it sets update laws, and `payoff(namespace, k)` returns
    the payoff of option `k` as a symbolic expression, where `namespace.target` holds the states and parameters of the
    group. With a `schedule` it sets one rule that walks the agents of the group in that order, and `payoff(agent,
    system, k)` returns the payoff as a number, so it can read the agent and the history and storage of the system view.
    Its scores may then be the components of one vector state, one per option, and a group of one member holds scores
    that several groups share. With `chosen`, the name of a state that holds the index of the option a member chose,
    only the score of that option moves, and the payoff of the other options is not read. Every agent reinforces at
    once under the schedule `'synchronous'`, one after the other in their order or in a new random order every step
    under `'sequential'` or `'random-sequential'`, and each in its own activation under `'member'`, together with the
    rules of its element and the other member interactions, in the order the system lists them, such as after a choice
    and a transfer in one visit.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element or group whose members hold the scores.

        scores:
            The names of the states that hold the score of every option, in the order of the options, or with a
            schedule the name of one vector state whose components are the scores.

        payoff:
            The function `payoff(namespace, k)` without a schedule, or `payoff(agent, system, k)` with one, that
            returns the realized payoff of option `k`.

        rate:
            The rate `a` at which the scores move towards the payoffs, between 0 and 1, by the rule `'average'`.

        rule:
            `'average'` to move the scores towards the payoffs, or `'cumulative'` to add the payoffs to the scores.

        phase:
            The phase of the step the rule of a schedule runs in, or `None` for the default phase.

        chosen:
            The name of the state that holds the index of the option a member chose, so only its score moves, or
            `None` to move every score.

        schedule:
            The order in which a rule walks the agents, `'synchronous'` for every agent at once, `'sequential'` or
            `'random-sequential'` for one agent after the other, or `'member'` for every agent in its own activation,
            or `None` for update laws of the scores of every member.

    Raises:
        ValidationError:
            The rule or the schedule is unknown, there are no scores, or the payoff is not callable.

    Examples:
        >>> from skcomplex.systems.economics.common.learning import PayoffLearningInteraction
        >>> reinforcement = PayoffLearningInteraction(
        ...     'reinforcement', target='learners', scores=('first', 'second'), payoff=lambda s, k: k, rate=0.5
        ... )
        >>> reinforcement.rate
        0.5
        >>> PayoffLearningInteraction(
        ...     'virtual', target='players', scores='scores', payoff=lambda a, m, k: k, rule='cumulative',
        ...     schedule='synchronous',
        ... ).rule
        'cumulative'
        >>> PayoffLearningInteraction(
        ...     'satisfaction', target='customers', scores=('s1', 's2'), payoff=lambda a, m, k: 1.0, chosen='shop',
        ...     schedule='member',
        ... ).chosen
        'shop'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        scores: str | Sequence[str],
        payoff: Callable[..., float | Expression],
        rate: Annotated[float, RealSet.closed(0, 1)] = 0.1,
        rule: str = 'average',
        phase: str | None = None,
        chosen: str | None = None,
        schedule: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.scores = scores
        self.payoff = payoff
        self.rate = rate
        self.rule = rule
        self.chosen = chosen
        self.phase = phase
        self.schedule = schedule

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update of the scores, or the rule of the agents, as the laws.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the members it binds.

        Raises:
            ValidationError:
                The rule or the schedule is unknown, there are no scores, or the payoff is not callable.
        """
        super()._prepare(context)
        self._check()
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = self._get_learning_laws(context)

    def _check(self) -> None:
        """Raise an error for an unknown variant, no scores, or an uncallable payoff."""
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r} of {self.label!r}. Use one of {SCHEDULES}.'
            raise ValidationError(error_msg)
        if self.rule not in RULES:
            error_msg = f'Unknown rule {self.rule!r} of {self.label!r}. Use one of {RULES}.'
            raise ValidationError(error_msg)
        vector = isinstance(self.scores, str) and bool(self.scores) and self.schedule is not None
        if not vector and (isinstance(self.scores, str) or not self.scores):
            error_msg = (
                f'{self.label!r} has the scores {self.scores!r}. Give a sequence of state names, one per option, or '
                'with a schedule the name of a vector state.'
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
        rate = 1.0 if self.rule == 'cumulative' else self.rate
        change = np.zeros_like(old)
        for k in options:
            payoff = float(self.payoff(agent, system, k))
            change[k] = payoff if self.rule == 'cumulative' else rate * (payoff - old[k])
        return old + change

    def _set_value(self, agent: AgentViewProtocol, value: object) -> None:
        """Set the scores of an agent."""
        scores = np.asarray(value, dtype=float)
        if isinstance(self.scores, str):
            setattr(agent, self.scores, scores)
            return
        for name, score in zip(self.scores, scores, strict=True):
            setattr(agent, name, float(score))

    def _update_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Move the scores of one agent, in its own activation."""
        if agent.alive:
            self._set_value(agent, self._get_scores(agent, system))

    def update(self, system: SystemViewProtocol) -> None:
        """Move the scores of every agent by the schedule.

        Under `'synchronous'` every agent reads the scores at the start of the rule, and otherwise the agents move one
        after the other, in their order or in a new random order.

        Args:
            system:
                The system view of the learning.
        """
        agents = list(system.get_agents('target'))
        if self.schedule == 'synchronous':
            values = [self._get_scores(agent, system) for agent in agents]
            for agent, value in zip(agents, values, strict=True):
                self._set_value(agent, value)
            return
        order = system.rng.permutation(len(agents)) if self.schedule == 'random-sequential' else range(len(agents))
        for index in order:
            self._update_member(agents[int(index)], system)

    def _get_learning_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of every score towards its payoff, or the rule of the schedule."""
        if self.schedule == 'member':
            return [UpdateLaw(namespace.target, self._update_member, phase=self.phase)]
        if self.schedule is not None:
            return [UpdateLaw(namespace, self.update, phase=self.phase)]
        laws: list[BaseLaw] = []
        for index, name in enumerate(self.scores):
            score = getattr(namespace.target, name)
            payoff = self.payoff(namespace, index)
            change = payoff if self.rule == 'cumulative' else namespace.rate * (payoff - score)
            if self.chosen is not None:
                change = where(getattr(namespace.target, self.chosen) == index, change, 0)
            laws.append(UpdateLaw(score, score + change))
        return laws
