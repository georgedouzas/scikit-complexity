"""Define choice interactions, which set a discrete state to an option drawn from the utilities of the options."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet
from scipy.special import expit, softmax

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..base import BasePart
from ..laws import RuleLaw, UpdateLaw
from ..laws.utils import categorical, uniform, where
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

RULES = ('logit', 'metropolis', 'fermi', 'best')
SCHEDULES = ('synchronous', 'random-sequential')
FORMS = ('symbolic', 'agent')
MIN_OPTIONS = 2


class ChoiceInteraction(Interaction):
    """Choice of every member of a group among options, drawn every step from the utilities of the options by a rule.

    The rule `'logit'`, which is the heat-bath rule, draws option `k` of utility `u_k` with probability
    `exp(beta u_k) / sum_j exp(beta u_j)`, where `beta` is the intensity of choice. The rules `'metropolis'` and
    `'fermi'` propose one of the other options uniformly and accept it with probability
    `min(1, exp(beta (u_new - u_old)))` or `1 / (1 + exp(-beta (u_new - u_old)))`. The rule `'best'` draws uniformly
    among the options of largest utility. Every step, each member revises its choice with probability `activity` and
    keeps it otherwise.

    In the `'symbolic'` form, `utility(namespace, option)` returns the utility as a symbolic expression, where
    `namespace.target` holds the states and parameters of the group, and every member draws at once. In the `'agent'`
    form, the group is an agent group and `utility(agent, model, option)` returns the utility as a number, so it can
    read the agent, its neighbours, and the history and storage of the model view. Its members draw at once, from the
    choices of the last step, under the `'synchronous'` schedule, and one after the other in a random order, each
    seeing the choices made before it, under the `'random-sequential'` schedule. An option of utility `-inf` is never
    drawn in the agent form.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose members choose.

        state:
            The name of the discrete state of the members that holds the chosen option.

        options:
            The values the state can take, at least two.

        utility:
            The function `utility(namespace, option)` in the symbolic form, or `utility(agent, model, option)` in the
            agent form, that returns the utility of an option.

        rule:
            The rule of the draw, `'logit'`, `'metropolis'`, `'fermi'`, or `'best'`.

        intensity:
            The intensity of choice `beta`, zero for uniform choice.

        schedule:
            The order in which the members revise, `'synchronous'` or, in the agent form, `'random-sequential'`.

        activity:
            The probability that a member revises its choice in a step.

        form:
            `'symbolic'` for update laws of the states of a group, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The rule, the schedule, or the form is unknown, the schedule is random-sequential in the symbolic form,
            there are fewer than two options, or the utility is not callable.

    Examples:
        >>> from skcomplex.core.interactions import ChoiceInteraction
        >>> from skcomplex.core.laws.utils import mean
        >>> choice = ChoiceInteraction(
        ...     'choice', target='spins', state='spin', options=(-1, 1), utility=lambda s, o: o * mean(s.target.spin)
        ... )
        >>> choice.check().rule
        'logit'
        >>> agents = ChoiceInteraction(
        ...     'choice', target='traders', state='plan', options=(0, 1), utility=lambda a, m, o: o, form='agent'
        ... )
        >>> type(agents.check().prepare().run(SimpleNamespace()).laws_[0]).__name__
        'RuleLaw'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        options: Sequence[float],
        utility: Callable[..., float | Expression],
        rule: str = 'logit',
        intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        schedule: str = 'synchronous',
        activity: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.options = options
        self.utility = utility
        self.rule = rule
        self.intensity = intensity
        self.schedule = schedule
        self.activity = activity
        self.form = form
        self.phase = phase

    def _check_variants(self) -> None:
        """Raise an error for an unknown rule, schedule, or form, or a random-sequential symbolic choice."""
        for name, value, variants in (
            ('rule', self.rule, RULES),
            ('schedule', self.schedule, SCHEDULES),
            ('form', self.form, FORMS),
        ):
            if value not in variants:
                error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {variants}.'
                raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.schedule != 'synchronous':
            error_msg = (
                f'{self.label!r} has the schedule {self.schedule!r}, which only the agent form runs. Use '
                "form='agent' or schedule='synchronous'."
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown variant, fewer than two options, or a utility that cannot be called.

        Raises:
            ValidationError:
                The type declares states, the rule, the schedule, or the form is unknown, the schedule is
                random-sequential in the symbolic form, there are fewer than two options, or the utility is not
                callable.
        """
        super()._check()
        self._check_variants()
        if len(self.options) < MIN_OPTIONS:
            error_msg = f'{self.label!r} chooses among {list(self.options)}. Give at least {MIN_OPTIONS} options.'
            raise ValidationError(error_msg)
        if not callable(self.utility):
            error_msg = f'The utility of {self.label!r} must be callable, but it is {self.utility!r}.'
            raise ValidationError(error_msg)

    def _select(self, index: Expression, values: Sequence[Expression]) -> Expression:
        """Return the value at a symbolic index."""
        return sum((where(index == position, value, 0) for position, value in enumerate(values)), SR(0))

    def _draw_softmax(self, state: Expression, utilities: Sequence[Expression], intensity: Expression) -> Expression:
        """Return the option drawn with probabilities proportional to `exp(beta u_k)`."""
        index = categorical(f'{state}_option', [exp(intensity * utility) for utility in utilities])
        return self._select(index, [SR(option) for option in self.options])

    def _draw_best(self, state: Expression, utilities: Sequence[Expression]) -> Expression:
        """Return an option drawn uniformly among the options of largest utility."""
        largest = utilities[0]
        for utility in utilities[1:]:
            largest = where(utility > largest, utility, largest)
        index = categorical(f'{state}_best', [where(utility >= largest, 1, 0) for utility in utilities])
        return self._select(index, [SR(option) for option in self.options])

    def _draw_proposal(self, state: Expression, utilities: Sequence[Expression], intensity: Expression) -> Expression:
        """Return a uniform proposal of another option, accepted by the Metropolis or the Fermi probability."""
        index = categorical(f'{state}_proposal', [where(state == option, 0, 1) for option in self.options])
        proposal = self._select(index, [SR(option) for option in self.options])
        gain = self._select(index, utilities) - sum(
            (where(state == option, utility, 0) for option, utility in zip(self.options, utilities, strict=True)),
            SR(0),
        )
        if self.rule == 'metropolis':
            acceptance = where(gain >= 0, 1, exp(intensity * gain))
        else:
            acceptance = 1 / (1 + exp(-intensity * gain))
        return where(uniform(f'{state}_acceptance') < acceptance, proposal, state)

    def _draw_symbolic(self, namespace: SimpleNamespace) -> Expression:
        """Return the option a member draws, as a symbolic expression of the namespace."""
        state = getattr(namespace.target, self.state)
        utilities = [SR(self.utility(namespace, option)) for option in self.options]
        if self.rule == 'logit':
            return self._draw_softmax(state, utilities, namespace.intensity)
        if self.rule == 'best':
            return self._draw_best(state, utilities)
        return self._draw_proposal(state, utilities, namespace.intensity)

    def _get_utilities(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> np.ndarray:
        """Return the utility of every option for an agent."""
        return np.array([float(self.utility(agent, model, option)) for option in self.options])

    def _draw_agent_proposal(self, agent: AgentViewProtocol, model: ModelViewProtocol, intensity: float) -> float:
        """Return a uniform proposal of another option of an agent if accepted, or its current option."""
        current = float(getattr(agent, self.state))
        others = [option for option in self.options if option != current]
        proposal = float(others[int(model.rng.integers(len(others)))])
        before = float(self.utility(agent, model, current)) if current in self.options else 0.0
        gain = float(self.utility(agent, model, proposal)) - before
        if self.rule == 'metropolis':
            acceptance = 1.0 if gain >= 0 else math.exp(intensity * gain)
        else:
            acceptance = float(expit(intensity * gain))
        return proposal if model.rng.random() < acceptance else current

    def _draw_agent(self, agent: AgentViewProtocol, model: ModelViewProtocol, intensity: float) -> float:
        """Return the option an agent draws by the rule."""
        if self.rule in ('metropolis', 'fermi'):
            return self._draw_agent_proposal(agent, model, intensity)
        utilities = self._get_utilities(agent, model)
        if self.rule == 'best':
            best = np.flatnonzero(utilities == utilities.max())
            return float(self.options[int(best[model.rng.integers(len(best))])])
        scaled = np.where(np.isneginf(utilities), -np.inf, intensity * utilities)
        probabilities = softmax(scaled)
        return float(self.options[int(model.rng.choice(len(self.options), p=probabilities))])

    def _is_active(self, model: ModelViewProtocol, activity: float) -> bool:
        """Return whether a member revises its choice in this step."""
        return activity >= 1 or bool(model.rng.random() < activity)

    def choose(self, model: ModelViewProtocol) -> None:
        """Set the state of every agent that revises in this step to the option it draws, by the schedule.

        Args:
            model:
                The model view of the interaction.
        """
        agents = list(model.agents('target'))
        intensity, activity = model.parameter('intensity'), model.parameter('activity')
        if self.schedule == 'random-sequential':
            for index in model.rng.permutation(len(agents)):
                if self._is_active(model, activity):
                    setattr(agents[index], self.state, self._draw_agent(agents[index], model, intensity))
            return
        chosen = [
            (agent, self._draw_agent(agent, model, intensity)) for agent in agents if self._is_active(model, activity)
        ]
        for agent, option in chosen:
            setattr(agent, self.state, option)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of the state of every member that revises, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`, `namespace.intensity`, and
                `namespace.activity`.

        Returns:
            laws:
                The update law of the state, or the rule that runs the choice of the agents.
        """
        if self.form == 'agent':
            return [RuleLaw(self.choose, phase=self.phase)]
        state = getattr(namespace.target, self.state)
        revised = where(uniform(f'{state}_activity') < namespace.activity, self._draw_symbolic(namespace), state)
        return [UpdateLaw(state, revised)]
