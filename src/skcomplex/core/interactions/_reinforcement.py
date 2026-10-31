"""Define reinforcement interactions, which move the score of every option towards its realized payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..base import BasePart
from ..laws import RuleLaw, UpdateLaw
from ..types import ModelViewProtocol
from ._interaction import Interaction

FORMS = ('symbolic', 'agent')


class ReinforcementInteraction(Interaction):
    """Reinforcement of the scores of options, each moving towards the payoff its option realized in the step.

    Every step, the score `s_k` of option `k` of every member moves towards the realized payoff `P_k` at the rate `a`,
    `s_k <- s_k + a (P_k - s_k)`, so a score is an exponentially weighted mean of past payoffs, and the last payoff at
    the rate 1. In the `'symbolic'` form, `payoff(namespace, k)` returns the payoff of option `k` as a symbolic
    expression, where `namespace.target` holds the states and parameters of the group. In the `'agent'` form, the
    group is an agent group and `payoff(agent, model, k)` returns the payoff as a number, so it can read the agent and
    the history and storage of the model view.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose members hold the scores.

        scores:
            The names of the states that hold the score of every option, in the order of the options.

        payoff:
            The function `payoff(namespace, k)` in the symbolic form, or `payoff(agent, model, k)` in the agent form,
            that returns the realized payoff of option `k`.

        rate:
            The rate `a` at which the scores move towards the payoffs, between 0 and 1.

        form:
            `'symbolic'` for update laws of the states of a group, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The form is unknown, there are no scores, or the payoff is not callable.

    Examples:
        >>> from skcomplex.core.interactions import ReinforcementInteraction
        >>> reinforcement = ReinforcementInteraction(
        ...     'reinforcement', target='learners', scores=('first', 'second'), payoff=lambda s, k: k, rate=0.5
        ... )
        >>> reinforcement.check().rate
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        scores: Sequence[str],
        payoff: Callable[..., float | Expression],
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.scores = scores
        self.payoff = payoff
        self.rate = rate
        self.form = form
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for an unknown form, no scores, or a payoff that cannot be called.

        Raises:
            ValidationError:
                The type declares states, the form is unknown, there are no scores, or the payoff is not callable.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'Unknown form {self.form!r} of {self.label!r}. Use one of {FORMS}.'
            raise ValidationError(error_msg)
        if isinstance(self.scores, str) or not self.scores:
            error_msg = (
                f'{self.label!r} has the scores {self.scores!r}. Give a sequence of state names, one per option.'
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
        rate = model.parameter('rate')
        agents = list(model.agents('target'))
        payoffs = [[float(self.payoff(agent, model, index)) for index in range(len(self.scores))] for agent in agents]
        for agent, realized in zip(agents, payoffs, strict=True):
            for name, value in zip(self.scores, realized, strict=True):
                score = float(getattr(agent, name))
                setattr(agent, name, score + rate * (value - score))

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
            laws.append(UpdateLaw(score, score + namespace.rate * (self.payoff(namespace, index) - score)))
        return laws
