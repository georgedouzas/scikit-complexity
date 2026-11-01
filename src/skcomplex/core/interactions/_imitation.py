"""Define imitation of a discrete state of the neighbours, by a random neighbour, the majority, the best, or Fermi."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, JumpLaw, RuleLaw, UpdateLaw
from ..laws.utils import categorical, neighbors, uniform, where
from ..spaces import AdjacencyNeighborhood
from ..types import AgentViewProtocol, ModelViewProtocol
from ._neighbor import NeighborInteraction

RULES = ('random', 'majority', 'best', 'better', 'fermi')
FORMS = ('rate', 'probability', 'agent')


class ImitationInteraction(NeighborInteraction):
    """Imitation of a discrete state of the neighbours of every member, at a rate or with a probability per step.

    By the rule `'random'` a member copies a neighbour chosen at random, by `'majority'` it takes the option that more
    of its neighbours hold than any other, and by `'best'` it copies the neighbour whose payoff is largest, when that
    payoff is larger than its own. By `'better'` it picks a neighbour at random and copies it when the neighbour's
    payoff is larger than its own. By `'fermi'` it picks a neighbour at random and copies it with probability
    `1 / (1 + exp(-beta (P_j - P_i)))`, where `P_j` and `P_i` are the payoffs of the neighbour and of the member and
    `beta` is the intensity. A member without such a neighbour keeps its state. The rules read the neighbourhood of
    the space, the graph of a network or the neighbourhood of a lattice.

    In the `'rate'` form every member imitates at the rate `rate`, by jumps between every pair of options in continuous
    time. In the `'probability'` form every member imitates with probability `probability` in every step of a
    discrete-time system, all members at once in its `phase`, reading the states the phases before it set. In the
    `'agent'` form the target is an agent group, and every agent imitates its neighbours in the space with probability
    `probability` in its `phase`, all agents at once, so they read the states of the step before. The rule `'better'`
    runs in the agent form only.

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

    Raises:
        ValidationError:
            The rule or the form is unknown, or the options are not distinct.

    Examples:
        >>> from skcomplex.core.interactions import ImitationInteraction
        >>> ImitationInteraction('imitation', target='voters', rule='majority').check().options
        (0.0, 1.0)
        >>> ImitationInteraction('imitation', target='players', rule='fermi', form='probability').check().form
        'probability'
        >>> ImitationInteraction('imitation', target='traders', rule='better', form='agent', phase='learn').check().rule
        'better'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
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

    def _check(self) -> None:
        """Raise an error for an unknown rule or form, or options that are not distinct.

        Raises:
            ValidationError:
                The rule or the form is unknown, or the options are not distinct.
        """
        super()._check()
        if self.rule not in RULES:
            error_msg = f'The rule of imitation is {self.rule!r}. Use one of {", ".join(RULES)}.'
            raise ValidationError(error_msg)
        if self.form not in FORMS:
            error_msg = f'The form of imitation is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if self.rule == 'better' and self.form != 'agent':
            error_msg = f'The rule of imitation {self.rule!r} runs in the agent form only. Use form `agent`.'
            raise ValidationError(error_msg)
        if len(set(self.options)) < max(len(self.options), 2):
            error_msg = f'The options of imitation are {self.options!r}. Give at least two distinct values.'
            raise ValidationError(error_msg)

    def _get_neighborhood(self) -> AdjacencyNeighborhood | None:
        """Return the neighbourhood of the layer, or `None` for the neighbourhood of the space."""
        return None if self.layer is None else AdjacencyNeighborhood(layer=self.layer)

    def _get_fermi(self, member: SimpleNamespace, other: SimpleNamespace, intensity: Expression) -> Expression:
        """Return the Fermi probability that a member copies another of a different payoff."""
        return 1 / (1 + exp(-intensity * (getattr(other, self.payoff) - getattr(member, self.payoff))))

    def _get_counts(self, namespace: SimpleNamespace) -> dict[float, Expression]:
        """Return the number of the neighbours of the target that hold every option."""
        source, over = namespace.source, self._get_neighborhood()
        return {
            option: neighbors(where(getattr(source, self.state) == option, 1, 0), over=over) for option in self.options
        }

    def _get_majority(self, namespace: SimpleNamespace) -> dict[float, Expression]:
        """Return the indicator of every option that more of the neighbours of the target hold than any other."""
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
        """Return the option of the neighbour of largest payoff and whether that payoff is larger than the target's."""
        source, over = namespace.source, self._get_neighborhood()
        payoff = getattr(source, self.payoff)
        best = neighbors(getattr(source, self.state), over=over, reduce='argmax', weight=payoff)
        largest = neighbors(payoff, over=over, reduce='max')
        return best, where(largest > getattr(namespace.target, self.payoff), 1, 0)

    def _switches(self, value: Expression, rate: Expression, chosen: Expression) -> list[BaseLaw]:
        """Return the jumps of a state from every option to the chosen option, at a rate when it is chosen."""
        return [
            JumpLaw(rate * where(value == first, 1, 0) * where(chosen == second, 1, 0), {value: second - first})
            for first, second in itertools.permutations(self.options, 2)
        ]

    def _copies(self, member: SimpleNamespace, other: SimpleNamespace, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the copies of the state of the other end of an edge by a member, chosen among its neighbours."""
        value, rate = getattr(member, self.state), self._get_parameter_symbol('rate', namespace) / member.degree
        if self.rule == 'fermi':
            rate = rate * self._get_fermi(member, other, self._get_parameter_symbol('intensity', namespace))
        return self._switches(value, rate, getattr(other, self.state))

    def _get_jumps(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps of the state at the rate of imitation, in continuous time."""
        value = getattr(namespace.target, self.state)
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
        source, target = namespace.source, namespace.target
        return [*self._copies(target, source, namespace), *self._copies(source, target, namespace)]

    def _draw_copy(self, namespace: SimpleNamespace) -> Expression:
        """Return the next state of the target by the random or the Fermi rule, from the option it copies or its own."""
        source, target, value = namespace.source, namespace.target, getattr(namespace.target, self.state)
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
        """Return the update of the state of the target with the probability of imitation, in discrete time."""
        value = getattr(namespace.target, self.state)
        if self.rule in ('random', 'fermi'):
            return [UpdateLaw(value, self._draw_copy(namespace), phase=self.phase)]
        if self.rule == 'majority':
            leads = self._get_majority(namespace)
            imitated = value + sum((lead * (option - value) for option, lead in leads.items()), SR(0))
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
        if self.form == 'agent':
            return [RuleLaw(self._imitate, phase=self.phase)]
        if self.form == 'probability':
            return self._get_update(namespace)
        return self._get_jumps(namespace)

    def _choose(
        self,
        agent: AgentViewProtocol,
        neighbours: Sequence[AgentViewProtocol],
        model: ModelViewProtocol,
    ) -> float | None:
        """Return the option an agent copies from its neighbours by the rule, or `None` to keep its own."""
        rng = model.rng
        if self.rule == 'majority':
            counts = [sum(getattr(other, self.state) == option for other in neighbours) for option in self.options]
            leaders = np.flatnonzero(np.asarray(counts) == max(counts))
            return float(self.options[int(leaders[0])]) if len(leaders) == 1 else None
        if self.rule == 'random':
            return float(getattr(neighbours[int(rng.integers(len(neighbours)))], self.state))
        own = float(getattr(agent, self.payoff))
        if self.rule == 'best':
            payoffs = np.array([getattr(other, self.payoff) for other in neighbours], dtype=float)
            best = int(np.argmax(payoffs))
            return float(getattr(neighbours[best], self.state)) if payoffs[best] > own else None
        other = neighbours[int(rng.integers(len(neighbours)))]
        difference = float(getattr(other, self.payoff)) - own
        if self.rule == 'better':
            return float(getattr(other, self.state)) if difference > 0 else None
        fermi = 1 / (1 + np.exp(-self._get_parameter_value('intensity', model) * difference))
        return float(getattr(other, self.state)) if rng.random() < fermi else None

    def _imitate(self, model: ModelViewProtocol) -> None:
        """Let every agent imitate its neighbours with the probability of imitation, all agents at once."""
        agents = list(model.agents(self._get_role('target')))
        probability = self._get_parameter_value('probability', model)
        members = {agent.id for agent in agents}
        choices: list[float | None] = []
        for agent in agents:
            neighbours = [other for other in agent.neighbors() if other.id in members]
            if not neighbours or model.rng.random() >= probability:
                choices.append(None)
                continue
            choices.append(self._choose(agent, neighbours, model))
        for agent, choice in zip(agents, choices, strict=True):
            if choice is not None:
                setattr(agent, self.state, choice)
