"""Define imitation of a discrete state of the neighbours.

A member imitates a random neighbour, the majority, the best, or by the Fermi rule.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import BaseSpec, ParameterSpec, RoleSpec, Tags
from ...types import ElementKind
from ..laws import BaseLaw, JumpLaw, RuleLaw, UpdateLaw
from ..laws.utils import categorical, neighbors, uniform, where
from ..spaces import AdjacencyNeighborhood, BaseSpace
from ..types import AgentViewProtocol, SystemViewProtocol
from ._base import SCHEDULES
from ._neighbor import NeighborInteraction

RULES = ('random', 'majority', 'best', 'better', 'fermi')
FORMS = ('rate', 'probability', 'agent')
TIES = ('first', 'random')


class ImitationInteraction(NeighborInteraction):
    """Imitation of a discrete state of the neighbours of every member.

    The imitation happens at a rate or with a probability per step. By the rule `'random'` a member copies a neighbour
    chosen at random, by `'majority'` it takes the option that more of its neighbours hold than any other, and by
    `'best'` it copies the neighbour whose payoff is largest, when that payoff is larger than its own. By `'better'` it
    picks a neighbour at random and copies it when the neighbour's payoff is larger than its own. By `'fermi'` it picks
    a neighbour at random and copies it with probability `1 / (1 + exp(-beta (P_j - P_i)))`, where `P_j` and `P_i` are
    the payoffs of the neighbour and of the member and `beta` is the intensity. A member without such a neighbour keeps
    its state. The rules read the neighbourhood of the space, the graph of a network or the neighbourhood of a lattice.

    In the `'rate'` form every member imitates at the rate `rate`, by jumps between every pair of options in continuous
    time. In the `'probability'` form every member imitates with probability `probability` in every step of a
    discrete-time system, all members at once in its `phase`, reading the states the phases before it set. In the
    `'agent'` form the target is an agent group, and every agent imitates its neighbours in the space with probability
    `probability` in its `phase`. The agents imitate at once, reading the states of the step before, under the schedule
    `'synchronous'`, one after the other in their order or in a new random order every step under `'sequential'` or
    `'random-sequential'`, each reading the states the agents before it set, or each in its own activation under
    `'member'`. The rule `'better'` runs in the agent form only. Among neighbours of equal largest payoff, the rule
    `'best'` of the agent form copies the first under the ties `'first'` and one drawn uniformly under `'random'`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the nodes or sites.

        state:
            The name of the discrete state of the members.

        payoff:
            The name of the state that holds the payoff of the members, which the rules `'best'` and `'fermi'` read.

        options:
            The values the state takes.

        rate:
            The rate at which every member imitates, in the rate form.

        rule:
            `'random'`, `'majority'`, `'best'`, or `'fermi'`.

        intensity:
            The intensity `beta` of the Fermi rule.

        form:
            `'rate'` for imitation in continuous time, `'probability'` for imitation in discrete time, or `'agent'`
            for imitation of the neighbours of the agents of an agent group.

        probability:
            The probability that a member imitates in a step, in the probability form.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

        phase:
            The phase of the step the agent and the probability forms run in, or `None` for the default phase.

        schedule:
            The order of the agents in the agent form, `'synchronous'`, `'sequential'`, `'random-sequential'`, or
            `'member'`.

        ties:
            The neighbour the rule `'best'` of the agent form copies among ties, `'first'` or `'random'`.

    Raises:
        ValidationError:
            The rule, the form, the schedule, or the ties are unknown, a schedule or ties other than the defaults are
            given outside the agent form, or the options are not distinct.

    Examples:
        >>> from skcomplex.core.components.interactions import ImitationInteraction
        >>> ImitationInteraction('imitation', target='voters', rule='majority').options
        (0.0, 1.0)
        >>> ImitationInteraction('imitation', target='players', rule='fermi', form='probability').form
        'probability'
        >>> ImitationInteraction('imitation', target='traders', rule='better', form='agent', phase='learn').rule
        'better'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (JumpLaw, RuleLaw, UpdateLaw)

    draws = True

    def _get_tags(self) -> Tags:
        """Return the tags of the rules, the updates, or the jumps of the rate form.

        The updates draw with a probability.
        """
        form = self._get_argument('form')
        if form == 'agent':
            return self._get_variant_tags((RuleLaw,))
        if form == 'probability':
            return self._get_variant_tags((UpdateLaw,), stochastic=True)
        return self._get_variant_tags((JumpLaw,))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str = 'state',
        payoff: str = 'payoff',
        options: Sequence[float] = (0.0, 1.0),
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rule: str = 'random',
        intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        form: str = 'rate',
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        layer: str | None = None,
        phase: str | None = None,
        schedule: str = 'synchronous',
        ties: str = 'first',
    ) -> None:
        super().__init__(label, layer=layer)
        self.target = target
        self.state = state
        self.payoff = payoff
        self.options = options
        self.rate = rate
        self.rule = rule
        self.intensity = intensity
        self.form = form
        self.probability = probability
        self.phase = phase
        self.schedule = schedule
        self.ties = ties

    def _check_agent_options(self) -> None:
        """Raise an error for an unknown schedule or ties, or non-default ones off agents.

        It also raises it for either of them not default outside the agent form.
        """
        self._check_variant('schedule', self.schedule, SCHEDULES)
        self._check_variant('ties', self.ties, TIES)
        if self.form != 'agent' and (self.schedule != 'synchronous' or self.ties != 'first'):
            error_msg = (
                f'The imitation {self.label!r} has the schedule {self.schedule!r} and the ties {self.ties!r}, which '
                'only the agent form runs. Use form="agent", or schedule="synchronous" and ties="first".'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for unknown rule, form, schedule, or ties, or repeated options.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The rule, the form, the schedule, or the ties are unknown, a schedule or ties other than the defaults
                are given outside the agent form, or the options are not distinct.
        """
        super()._prepare(context)
        self._check_variant('rule', self.rule, RULES, (BaseSpec,))
        if self.form not in FORMS:
            error_msg = f'The form of imitation is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if self.rule == 'better' and self.form != 'agent':
            error_msg = f'The rule of imitation {self.rule!r} runs in the agent form only. Use form `agent`.'
            raise ValidationError(error_msg)
        self._check_agent_options()
        if len(set(self.options)) < max(len(self.options), 2):
            error_msg = f'The options of imitation are {self.options!r}. Give at least two distinct values.'
            raise ValidationError(error_msg)

    def _get_neighborhood(self) -> AdjacencyNeighborhood | None:
        """Return the neighbourhood of the layer.

        It returns `None` for the neighbourhood of the space.
        """
        return None if self.layer is None else AdjacencyNeighborhood(layer=self.layer)

    def _get_fermi(self, member: SimpleNamespace, other: SimpleNamespace, intensity: Expression) -> Expression:
        """Return the Fermi probability that a member copies another of a different payoff."""
        return 1 / (1 + exp(-intensity * (getattr(other, self.payoff) - getattr(member, self.payoff))))

    def _get_counts(self, namespace: SimpleNamespace) -> dict[float, Expression]:
        """Return the number of the neighbours of the target that hold every option."""
        source, over = self._get_member(namespace, 'source'), self._get_neighborhood()
        return {
            option: neighbors(where(getattr(source, self.state) == option, 1, 0), over=over) for option in self.options
        }

    def _get_majority(self, namespace: SimpleNamespace) -> dict[float, Expression]:
        """Return the indicator of every option most neighbours of the target hold.

        More of the neighbours hold that option than any other.
        """
        counts = self._get_counts(namespace)
        leads = {}
        for option, count in counts.items():
            lead = SR(1)
            for other, other_count in counts.items():
                if other != option:
                    lead = lead * where(count > other_count, 1, 0)
            leads[option] = lead
        return leads

    def _get_best(self, namespace: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the option of the neighbour of largest payoff and if it beats the target.

        It returns whether that payoff is larger than the target's.
        """
        source, over = self._get_member(namespace, 'source'), self._get_neighborhood()
        payoff = getattr(source, self.payoff)
        best = neighbors(getattr(source, self.state), over=over, reduce='argmax', weight=payoff)
        largest = neighbors(payoff, over=over, reduce='max')
        return best, where(largest > getattr(self._get_member(namespace, 'target'), self.payoff), 1, 0)

    def _switches(self, value: Expression, rate: Expression, chosen: Expression) -> list[BaseLaw]:
        """Return the jumps of a state from every option to the chosen option.

        A jump happens at a rate when the option is chosen.
        """
        return [
            JumpLaw(rate * where(value == first, 1, 0) * where(chosen == second, 1, 0), {value: second - first})
            for first, second in itertools.permutations(self.options, 2)
        ]

    def _copies(self, member: SimpleNamespace, other: SimpleNamespace, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the copies by a member of the state of the other end of an edge.

        The other end is chosen among its neighbours.
        """
        value, rate = getattr(member, self.state), self._get_parameter_symbol('rate', namespace) / member.degree
        if self.rule == 'fermi':
            rate = rate * self._get_fermi(member, other, self._get_parameter_symbol('intensity', namespace))
        return self._switches(value, rate, getattr(other, self.state))

    def _get_jumps(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps of the state at the rate of imitation, in continuous time."""
        value = getattr(self._get_member(namespace, 'target'), self.state)
        if self.rule == 'majority':
            leads = self._get_majority(namespace)
            return [
                JumpLaw(
                    self._get_parameter_symbol('rate', namespace) * where(value == first, 1, 0) * leads[second],
                    {value: second - first},
                )
                for first, second in itertools.permutations(self.options, 2)
            ]
        if self.rule == 'best':
            best, better = self._get_best(namespace)
            return self._switches(value, self._get_parameter_symbol('rate', namespace) * better, best)
        if self.rule not in ('random', 'fermi'):
            raise self._get_variant_error(self.rule, 'rule', '_get_jumps')
        source, target = self._get_member(namespace, 'source'), self._get_member(namespace, 'target')
        return [*self._copies(target, source, namespace), *self._copies(source, target, namespace)]

    def _draw_copy(self, namespace: SimpleNamespace) -> Expression:
        """Return the next state of the target by the random or the Fermi rule.

        It comes from the option it copies or its own.
        """
        source, target = self._get_member(namespace, 'source'), self._get_member(namespace, 'target')
        value = getattr(target, self.state)
        probability, over = self._get_parameter_symbol('probability', namespace), self._get_neighborhood()
        counts = self._get_counts(namespace)
        degree = sum(counts.values(), SR(0))
        if self.rule == 'fermi':
            fermi = self._get_fermi(target, source, self._get_parameter_symbol('intensity', namespace))
            weights = [
                probability * neighbors(where(getattr(source, self.state) == option, 1, 0) * fermi, over=over)
                for option in self.options
            ]
        else:
            weights = [probability * counts[option] for option in self.options]
        stay = degree - sum(weights, SR(0)) + where(degree == 0, 1, 0)
        index = categorical(f'{value}_imitation', [stay, *weights])
        return value + sum(
            (where(index == position + 1, option - value, 0) for position, option in enumerate(self.options)),
            SR(0),
        )

    def _get_update(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the state of the target with the probability of imitation.

        It is for discrete time.
        """
        value = getattr(self._get_member(namespace, 'target'), self.state)
        if self.rule in ('random', 'fermi'):
            return [UpdateLaw(value, self._draw_copy(namespace), phase=self.phase)]
        if self.rule == 'majority':
            leads = self._get_majority(namespace)
            imitated = value + sum((lead * (option - value) for option, lead in leads.items()), SR(0))
        elif self.rule != 'best':
            raise self._get_variant_error(self.rule, 'rule', '_get_update')
        else:
            best, better = self._get_best(namespace)
            imitated = where(better == 1, best, value)
        return [
            UpdateLaw(
                value,
                where(
                    uniform(f'{value}_imitation') < self._get_parameter_symbol('probability', namespace),
                    imitated,
                    value,
                ),
                phase=self.phase,
            ),
        ]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps or the update of the state by the rule of the imitation.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The jumps of the state between every pair of options, or the update of the state of every member.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._imitate_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self._imitate, phase=self.phase)]
        if self.form == 'probability':
            return self._get_update(namespace)
        return self._get_jumps(namespace)

    def _choose(
        self,
        agent: AgentViewProtocol,
        neighbours: Sequence[AgentViewProtocol],
        system: SystemViewProtocol,
    ) -> float | None:
        """Return the option an agent copies from its neighbours by the rule.

        It returns `None` for the agent to keep its own.
        """
        rng = system.rng
        if self.rule == 'majority':
            counts = [sum(getattr(other, self.state) == option for other in neighbours) for option in self.options]
            leaders = np.flatnonzero(np.asarray(counts) == max(counts))
            return float(self.options[int(leaders[0])]) if len(leaders) == 1 else None
        if self.rule == 'random':
            return float(getattr(neighbours[int(rng.integers(len(neighbours)))], self.state))
        own = float(getattr(agent, self.payoff))
        if self.rule == 'best':
            payoffs = np.array([getattr(other, self.payoff) for other in neighbours], dtype=float)
            best = self._pick_best(payoffs, self.ties, system)
            return float(getattr(neighbours[best], self.state)) if payoffs[best] > own else None
        other = neighbours[int(rng.integers(len(neighbours)))]
        difference = float(getattr(other, self.payoff)) - own
        if self.rule == 'better':
            return float(getattr(other, self.state)) if difference > 0 else None
        if self.rule != 'fermi':
            raise self._get_variant_error(self.rule, 'rule', '_choose')
        fermi = 1 / (1 + np.exp(-self._get_parameter_value('intensity', system) * difference))
        return float(getattr(other, self.state)) if rng.random() < fermi else None

    def _get_choice(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float | None:
        """Return the option an agent copies with the probability of imitation.

        It returns `None` for the agent to keep its own.
        """
        groups = set(system.groups(self._get_role('target')))
        neighbours = [other for other in agent.neighbors(layer=self.layer) if other.group in groups]
        if not neighbours or system.rng.random() >= self._get_parameter_value('probability', system):
            return None
        return self._choose(agent, neighbours, system)

    def _set_choice(self, agent: AgentViewProtocol, choice: float | None) -> None:
        """Set the state of an agent to the option it copies, if any."""
        if choice is not None:
            setattr(agent, self.state, choice)

    def _imitate_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Let one agent imitate its neighbours with the probability of imitation.

        This happens in its own activation.
        """
        if agent.alive:
            self._set_choice(agent, self._get_choice(agent, system))

    def _imitate(self, system: SystemViewProtocol) -> None:
        """Let every agent imitate its neighbours with the probability of imitation.

        The agents follow the schedule.
        """
        self._update_agents(system, self.schedule, self._get_choice, self._set_choice)
