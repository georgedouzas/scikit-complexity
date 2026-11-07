"""Define reinforcement interactions, which move the score of every option towards its realized payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec, Tags
from ...types import Array, ElementKind
from ..base import BasePart
from ..laws import BaseLaw, RuleLaw, UpdateLaw
from ..laws.utils import where
from ..types import AgentViewProtocol, SystemViewProtocol
from ._base import SCHEDULES
from ._interaction import Interaction

FORMS = ('symbolic', 'agent')
RULES = ('average', 'cumulative')


class ReinforcementInteraction(Interaction):
    """Reinforcement of the scores of options, each moving towards the payoff its option realized in the step.

    Every step, the score `s_k` of option `k` of every member moves towards the realized payoff `P_k` at the rate `a`,
    `s_k <- s_k + a (P_k - s_k)`, so a score is an exponentially weighted mean of past payoffs, and the last payoff at
    the rate 1, by the rule `'average'`. By the rule `'cumulative'` the score adds the payoff, `s_k <- s_k + P_k`, so
    it is the sum of past payoffs, such as a virtual score. In the agent form the scores may be the components of one
    vector state, one per option, and a group of one member holds scores that several groups share. In the
    `'symbolic'` form, `payoff(namespace, k)` returns the payoff of option `k` as a symbolic
    expression, where `namespace.target` holds the states and parameters of the group. In the `'agent'` form, the
    group is an agent group and `payoff(agent, system, k)` returns the payoff as a number, so it can read the agent and
    the history and storage of the system view. With `chosen`, the name of a state that holds the index of the option a
    member chose, only the score of that option moves, and the payoff of the other options is not read. In the agent
    form every agent reinforces at once under the schedule `'synchronous'`, one after the other in their order or in a
    new random order every step under `'sequential'` or `'random-sequential'`, and each in its own activation under
    `'member'`, together with the rules of its element and the other member interactions, in the order the system
    lists them, such as after a choice and a transfer in one visit.

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
        >>> from skcomplex.core.components.interactions import ReinforcementInteraction
        >>> reinforcement = ReinforcementInteraction(
        ...     'reinforcement', target='learners', scores=('first', 'second'), payoff=lambda s, k: k, rate=0.5
        ... )
        >>> reinforcement.rate
        0.5
        >>> ReinforcementInteraction(
        ...     'virtual', target='players', scores='scores', payoff=lambda a, m, k: k, rule='cumulative', form='agent'
        ... ).rule
        'cumulative'
        >>> ReinforcementInteraction(
        ...     'satisfaction', target='customers', scores=('s1', 's2'), payoff=lambda a, m, k: 1.0, chosen='shop',
        ...     schedule='member', form='agent',
        ... ).chosen
        'shop'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rules of the agent form or of the updates of the symbolic reinforcement."""
        return self._get_variant_tags((RuleLaw,) if self._get_argument('form') == 'agent' else (UpdateLaw,))

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
        super().__init__(label)
        self.target = target
        self.scores = scores
        self.payoff = payoff
        self.rate = rate
        self.rule = rule
        self.form = form
        self.phase = phase
        self.chosen = chosen
        self.schedule = schedule

    def _check(self) -> None:
        """Raise an error for an unknown variant, no scores, or a payoff that cannot be called.

        Raises:
            ValidationError:
                The type declares states, the form, the rule, or the schedule is unknown, the symbolic form runs member
                by member, there are no scores, or the payoff is not callable.
        """
        super()._check()
        for name, value, variants in (
            ('form', self.form, FORMS),
            ('rule', self.rule, RULES),
            ('schedule', self.schedule, SCHEDULES),
        ):
            if value not in variants:
                error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {variants}.'
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

    def _read(self, agent: AgentViewProtocol) -> Array:
        """Return the scores of an agent, one per option."""
        if isinstance(self.scores, str):
            return np.array(getattr(agent, self.scores), dtype=float).ravel()
        return np.array([getattr(agent, name) for name in self.scores], dtype=float)

    def _get_scores(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> Array:
        """Return the new scores of an agent, every score or only that of its chosen option moved by its payoff."""
        old = self._read(agent)
        options = range(len(old)) if self.chosen is None else [int(getattr(agent, self.chosen))]
        rate = 1.0 if self.rule == 'cumulative' else self._get_parameter_value('rate', system)
        change = np.zeros_like(old)
        for k in options:
            payoff = float(self.payoff(agent, system, k))
            change[k] = payoff if self.rule == 'cumulative' else rate * (payoff - old[k])
        return old + change

    def _write(self, agent: AgentViewProtocol, scores: Array) -> None:
        """Set the scores of an agent."""
        if isinstance(self.scores, str):
            setattr(agent, self.scores, scores)
            return
        for name, value in zip(self.scores, scores, strict=True):
            setattr(agent, name, float(value))

    def _reinforce_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Move the scores of one agent towards its payoffs, in its own activation."""
        if agent.alive:
            self._write(agent, self._get_scores(agent, system))

    def reinforce(self, system: SystemViewProtocol) -> None:
        """Move the scores of every agent towards the payoffs of its options, by the schedule.

        Args:
            system:
                The system view of the interaction.
        """
        self._update_agents(system, self.schedule, self._get_scores, self._write)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of every score towards its payoff, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.rate`.

        Returns:
            laws:
                The update laws of the scores, or the rule that reinforces the scores of the agents.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._reinforce_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self.reinforce, phase=self.phase)]
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
