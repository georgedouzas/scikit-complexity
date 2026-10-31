"""Define reinforcement interactions, which move the score of every option towards its realized payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ...types import Array
from ..base import BasePart
from ..laws import RuleLaw, UpdateLaw
from ..types import AgentViewProtocol, ModelViewProtocol
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
    group is an agent group and `payoff(agent, model, k)` returns the payoff as a number, so it can read the agent and
    the history and storage of the model view.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose members hold the scores.

        scores:
            The names of the states that hold the score of every option, in the order of the options, or in the agent
            form the name of one vector state whose components are the scores.

        payoff:
            The function `payoff(namespace, k)` in the symbolic form, or `payoff(agent, model, k)` in the agent form,
            that returns the realized payoff of option `k`.

        rate:
            The rate `a` at which the scores move towards the payoffs, between 0 and 1, by the rule `'average'`.

        rule:
            `'average'` to move the scores towards the payoffs, or `'cumulative'` to add the payoffs to the scores.

        form:
            `'symbolic'` for update laws of the states of a group, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The form or the rule is unknown, there are no scores, or the payoff is not callable.

    Examples:
        >>> from skcomplex.core.interactions import ReinforcementInteraction
        >>> reinforcement = ReinforcementInteraction(
        ...     'reinforcement', target='learners', scores=('first', 'second'), payoff=lambda s, k: k, rate=0.5
        ... )
        >>> reinforcement.check().rate
        0.5
        >>> ReinforcementInteraction(
        ...     'virtual', target='players', scores='scores', payoff=lambda a, m, k: k, rule='cumulative', form='agent'
        ... ).check().rule
        'cumulative'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        scores: str | Sequence[str],
        payoff: Callable[..., float | Expression],
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        rule: str = 'average',
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.scores = scores
        self.payoff = payoff
        self.rate = rate
        self.rule = rule
        self.form = form
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for an unknown form or rule, no scores, or a payoff that cannot be called.

        Raises:
            ValidationError:
                The type declares states, the form or the rule is unknown, there are no scores, or the payoff is not
                callable.
        """
        super()._check()
        for name, value, variants in (('form', self.form, FORMS), ('rule', self.rule, RULES)):
            if value not in variants:
                error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {variants}.'
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

    def reinforce(self, model: ModelViewProtocol) -> None:
        """Move the scores of every agent towards the payoffs of its options, all agents at once.

        Args:
            model:
                The model view of the interaction.
        """
        rate = 1.0 if self.rule == 'cumulative' else model.parameter('rate')
        agents = list(model.agents('target'))
        scores = [self._read(agent) for agent in agents]
        payoffs = [
            np.array([self.payoff(agent, model, k) for k in range(len(old))], float)
            for agent, old in zip(agents, scores, strict=True)
        ]
        for agent, old, realized in zip(agents, scores, payoffs, strict=True):
            new = old + realized if self.rule == 'cumulative' else old + rate * (realized - old)
            if isinstance(self.scores, str):
                setattr(agent, self.scores, new)
            else:
                for name, value in zip(self.scores, new, strict=True):
                    setattr(agent, name, float(value))

    def _read(self, agent: AgentViewProtocol) -> Array:
        """Return the scores of an agent, one per option."""
        if isinstance(self.scores, str):
            return np.array(getattr(agent, self.scores), dtype=float).ravel()
        return np.array([getattr(agent, name) for name in self.scores], dtype=float)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of every score towards its payoff, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.rate`.

        Returns:
            laws:
                The update laws of the scores, or the rule that reinforces the scores of the agents.
        """
        if self.form == 'agent':
            return [RuleLaw(self.reinforce, phase=self.phase)]
        laws: list[BasePart] = []
        for index, name in enumerate(self.scores):
            score = getattr(namespace.target, name)
            payoff = self.payoff(namespace, index)
            change = payoff if self.rule == 'cumulative' else namespace.rate * (payoff - score)
            laws.append(UpdateLaw(score, score + change))
        return laws
