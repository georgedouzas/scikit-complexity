"""Define the interaction by which every member chooses one option by its utility."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet
from scipy.special import expit, softmax

from ....core.components.base import BasePart
from ....core.components.interactions import SCHEDULES, Interaction
from ....core.components.laws import BaseLaw, RuleLaw, UpdateLaw
from ....core.components.laws.utils import categorical, uniform, where
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import BaseSpec, ParameterSpec, RoleSpec, Tags
from ....core.types import ElementKind

RULES = ('logit', 'metropolis', 'fermi', 'best')
FORMS = ('symbolic', 'agent')
PROPOSALS = ('others', 'all')
TIES = ('random', 'first')
MIN_OPTIONS = 2


class ChoiceInteraction(Interaction):
    """An interaction by which members of a group choose among options by utility.

    At every step each member revises its choice with the probability `activity` and keeps it otherwise. The `rule`
    draws the new option from the utilities `u_k` of the options:

    - `'logit'`: option `k` with probability `exp(beta u_k) / sum_j exp(beta u_j)`, where `beta` is the intensity of
      choice.
    - `'metropolis'` and `'fermi'`: a proposed option, accepted with probability `min(1, exp(beta (u_new - u_old)))` or
      `1 / (1 + exp(-beta (u_new - u_old)))`.
    - `'best'`: an option of largest utility, with ties broken by `ties`.

    In the symbolic form `utility(namespace, option)` returns an expression, and every member draws at once. In the
    agent form `utility(agent, system, option)` returns a number, the members draw by `schedule`, and an option of
    utility `-inf` is never drawn.

    Unlike `TransitionInteraction`, which moves a state between values by given rates, it draws the new value from the
    utilities of all options at once. Choose it when members pick the option that pays best, with noise.

    It appears in many fields. Commuters choose a mode of transport, consumers choose a brand, farmers choose a crop,
    and voters choose a party by a logit model of discrete choice.

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
            The function `utility(namespace, option)` in the symbolic form, or `utility(agent, system, option)` in the
            agent form, that returns the utility of an option.

        rule:
            The rule of the draw, `'logit'`, `'metropolis'`, `'fermi'`, or `'best'`.

        intensity:
            The intensity of choice `beta`, zero for uniform choice.

        schedule:
            The order in which the members revise, `'synchronous'` or, in the agent form, `'sequential'`,
            `'random-sequential'`, or `'member'`.

        activity:
            The probability that a member revises its choice in a step.

        form:
            `'symbolic'` for update laws of the states of a group, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

        proposal:
            The options the rules `'metropolis'` and `'fermi'` propose from, `'others'` for the options other than the
            current one, or `'all'` for every option.

        ties:
            The option the rule `'best'` takes among options of equal largest utility, `'random'` for one drawn
            uniformly or `'first'` for the first in the order of the options.

    Raises:
        ValidationError:
            The rule, the schedule, the form, the proposal, or the ties are unknown, the schedule is not synchronous in
            the symbolic form, there are fewer than two options, or the utility is not callable.

    Examples:
        >>> from skcomplex.systems.general.agents import ChoiceInteraction
        >>> from skcomplex.core.components.laws.utils import mean
        >>> choice = ChoiceInteraction(
        ...     'choice', target='spins', state='spin', options=(-1, 1), utility=lambda s, o: o * mean(s.target.spin)
        ... )
        >>> choice.rule
        'logit'
        >>> agents = ChoiceInteraction(
        ...     'choice', target='traders', state='plan', options=(0, 1), utility=lambda a, m, o: o, form='agent'
        ... )
        >>> type(agents.prepare().run(SimpleNamespace()).laws_[0]).__name__
        'RuleLaw'
        >>> ChoiceInteraction(
        ...     'heat-bath', target='spins', state='spin', options=(-1, 0, 1), utility=lambda s, o: o, rule='fermi',
        ...     proposal='all'
        ... ).proposal
        'all'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the agent-form rules or of the symbolic choice updates."""
        return self._get_variant_tags((RuleLaw,) if self._get_argument('form') == 'agent' else (UpdateLaw,))

    def _get_draws(self) -> bool:
        """Return whether the choice draws.

        Every rule draws but the best option with the first of ties every step.
        """
        fixed = self.rule == 'best' and self.ties == 'first' and self.activity == 1
        return super()._get_draws() or not fixed

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str,
        options: Sequence[float],
        utility: Callable[..., float | Expression],
        rule: str = 'logit',
        intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        schedule: str = 'synchronous',
        activity: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        form: str = 'symbolic',
        phase: str | None = None,
        proposal: str = 'others',
        ties: str = 'random',
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
        self.proposal = proposal
        self.ties = ties

    def _check_variants(self) -> None:
        """Raise an error for an unknown variant or a symbolic choice not synchronous."""
        self._check_variant('rule', self.rule, RULES, (BaseSpec,))
        for name, value, variants in (
            ('schedule', self.schedule, SCHEDULES),
            ('form', self.form, FORMS),
            ('proposal', self.proposal, PROPOSALS),
            ('ties', self.ties, TIES),
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

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown variant, fewer than two options, or a bad utility.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, the rule, the schedule, the form, the proposal, or the ties are unknown, the
                schedule is not synchronous in the symbolic form, there are fewer than two options, or the utility is
                not callable.
        """
        super()._prepare(context)
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
        weights = [where(utility >= largest, 1, 0) for utility in utilities]
        if self.ties == 'first':
            weights = [
                weight * where(sum(weights[:position], SR(0)) > 0, 0, 1) for position, weight in enumerate(weights)
            ]
        index = categorical(f'{state}_best', weights)
        return self._select(index, [SR(option) for option in self.options])

    def _draw_proposal(self, state: Expression, utilities: Sequence[Expression], intensity: Expression) -> Expression:
        """Return a uniform proposal of an option.

        The proposal is accepted by the Metropolis or the Fermi probability.
        """
        weights = [SR(1) if self.proposal == 'all' else where(state == option, 0, 1) for option in self.options]
        index = categorical(f'{state}_proposal', weights)
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
        state = getattr(self._get_member(namespace, 'target'), self.state)
        utilities = [SR(self.utility(namespace, option)) for option in self.options]
        if self.rule == 'logit':
            return self._draw_softmax(state, utilities, self._get_parameter_symbol('intensity', namespace))
        if self.rule == 'best':
            return self._draw_best(state, utilities)
        if self.rule in ('metropolis', 'fermi'):
            return self._draw_proposal(state, utilities, self._get_parameter_symbol('intensity', namespace))
        raise self._get_variant_error(self.rule, 'rule', '_draw_symbolic')

    def _get_utilities(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> np.ndarray:
        """Return the utility of every option for an agent."""
        return np.array([float(self.utility(agent, system, option)) for option in self.options])

    def _draw_agent_proposal(self, agent: AgentViewProtocol, system: SystemViewProtocol, intensity: float) -> float:
        """Return a uniform proposal of an option of an agent if accepted.

        Otherwise it returns its current option.
        """
        current = float(getattr(agent, self.state))
        others = [option for option in self.options if self.proposal == 'all' or option != current]
        proposal = float(others[int(system.rng.integers(len(others)))])
        before = float(self.utility(agent, system, current)) if current in self.options else 0.0
        gain = float(self.utility(agent, system, proposal)) - before
        if self.rule == 'metropolis':
            acceptance = 1.0 if gain >= 0 else math.exp(intensity * gain)
        else:
            acceptance = float(expit(intensity * gain))
        return proposal if system.rng.random() < acceptance else current

    def _draw_agent(self, agent: AgentViewProtocol, system: SystemViewProtocol, intensity: float) -> float:
        """Return the option an agent draws by the rule."""
        if self.rule in ('metropolis', 'fermi'):
            return self._draw_agent_proposal(agent, system, intensity)
        utilities = self._get_utilities(agent, system)
        if self.rule == 'best':
            return float(self.options[self._pick_best(utilities, self.ties, system)])
        if self.rule != 'logit':
            raise self._get_variant_error(self.rule, 'rule', '_draw_agent')
        scaled = np.where(np.isneginf(utilities), -np.inf, intensity * utilities)
        probabilities = softmax(scaled)
        return float(self.options[int(system.rng.choice(len(self.options), p=probabilities))])

    def _is_active(self, system: SystemViewProtocol, activity: float) -> bool:
        """Return whether a member revises its choice in this step."""
        return activity >= 1 or bool(system.rng.random() < activity)

    def _get_choice(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float | None:
        """Return the option an agent draws if it revises in this step.

        It returns `None` for the agent to keep its own.
        """
        if not self._is_active(system, self._get_parameter_value('activity', system)):
            return None
        return self._draw_agent(agent, system, self._get_parameter_value('intensity', system))

    def _set_choice(self, agent: AgentViewProtocol, option: float | None) -> None:
        """Set the state of an agent to the option it draws, if any."""
        if option is not None:
            setattr(agent, self.state, option)

    def _choose_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Set the state of one agent to the option it draws if it revises.

        This happens in its own activation.
        """
        if agent.alive:
            self._set_choice(agent, self._get_choice(agent, system))

    def choose(self, system: SystemViewProtocol) -> None:
        """Set the state of every revising agent to the option it draws, by the schedule.

        An agent revises in this step.

        Args:
            system:
                The system view of the interaction.
        """
        self._update_agents(system, self.schedule, self._get_choice, self._set_choice)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the update of the state of every revising member, or the agent-form rule.

        The agent form returns its rule instead.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`, `namespace.intensity`, and
                `namespace.activity`.

        Returns:
            laws:
                The update law of the state, or the rule that runs the choice of the agents.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._choose_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self.choose, phase=self.phase)]
        state = getattr(self._get_member(namespace, 'target'), self.state)
        revised = where(
            uniform(f'{state}_activity') < self._get_parameter_symbol('activity', namespace),
            self._draw_symbolic(namespace),
            state,
        )
        return [UpdateLaw(state, revised)]
