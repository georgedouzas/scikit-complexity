"""Define the interaction by which members copy a state of their neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Tags
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw, UpdateLaw
from ....core.components.laws.utils import categorical, neighbors, uniform, where
from ....core.components.spaces import AdjacencyNeighborhood, BaseSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.enums import ScheduleKind
from ....core.errors import ValidationError
from ....core.specs import BaseSpec
from ....core.types import Role

RULES = ('random', 'majority', 'best', 'better', 'fermi')
TIES = ('first', 'random')


class ImitationInteraction(Interaction):
    """An interaction by which members copy a discrete state from their neighbours.

    The `rule` says whom a member copies:

    - `'random'`: a random neighbour.
    - `'majority'`: the option that more of its neighbours hold than any other.
    - `'best'`: the neighbour with the largest payoff, when it beats its own.
    - `'better'`: a random neighbour, when its payoff beats its own, with a schedule only.
    - `'fermi'`: a random neighbour, with probability `1 / (1 + exp(-beta (P_j - P_i)))`, where `P_j` and `P_i` are the
      payoffs of the neighbour and the member.

    Without a `schedule` it sets symbolic laws: jumps at the rate `rate` in continuous time, or, when `probability` is
    given, updates with that probability per step for every member at once. With a `schedule` it sets one rule of the
    whole system that walks the agents in that order with its own draws. A member without such a neighbour keeps its
    state.

    Unlike a sum of `neighbors` in a `RateLaw`, which averages numeric states of neighbours, it copies one discrete
    value from them. Choose it when members adopt the choices of those around them.

    It appears in many fields. Voters copy their friends in the voter model, players copy successful strategies in
    evolutionary games on networks, and farmers adopt the crops of their neighbours.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group on the nodes or sites.

        state:
            The name of the discrete state of the members.

        payoff:
            The name of the state that holds the payoff of the members, which the rules `'best'` and `'fermi'` read.

        options:
            The values the state takes.

        rate:
            The rate at which every member imitates in continuous time.

        rule:
            `'random'`, `'majority'`, `'best'`, or `'fermi'`.

        intensity:
            The intensity `beta` of the Fermi rule.

        probability:
            The probability that a member imitates in a step, or `None` for imitation in continuous time without a
            schedule and for every agent in every step with one.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

        phase:
            The phase of the step the updates of a probability or the rule of a schedule run in, or `None` for the
            default phase.

        schedule:
            The order in which a rule walks the agents, `'synchronous'` for every agent at once, `'sequential'` or
            `'random-sequential'` for one agent after the other, or `'member'` for every agent in its own activation,
            or `None` for symbolic laws.

        ties:
            The neighbour the rule `'best'` copies among ties with a schedule, `'first'` or `'random'`.

    Raises:
        ValidationError:
            The rule, the schedule, or the ties are unknown, the rule `'better'` or ties other than `'first'` are given
            without a schedule, or the options are not distinct.

    Examples:
        >>> from skcomplex.systems.general.agents import ImitationInteraction
        >>> ImitationInteraction('imitation', target='voters', rule='majority').options
        (0.0, 1.0)
        >>> ImitationInteraction('imitation', target='players', rule='fermi', probability=0.5).probability
        0.5
        >>> ImitationInteraction('imitation', target='traders', rule='better', schedule='synchronous').rule
        'better'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (JumpLaw, UpdateLaw)

    draws = True

    def _get_tags(self) -> Tags:
        """Return the tags of the rule of a schedule, the updates of a probability, or the jumps of a rate.

        The updates draw with a probability.
        """
        if self._get_argument('schedule') is not None:
            return self._get_variant_tags((UpdateLaw,))
        if self._get_argument('probability') is not None:
            return self._get_variant_tags((UpdateLaw,), stochastic=True)
        return self._get_variant_tags((JumpLaw,))

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'state',
        payoff: str = 'payoff',
        options: Sequence[float] = (0.0, 1.0),
        rate: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        rule: str = 'random',
        intensity: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        probability: Annotated[float, RealSet.closed(0, 1)] | None = None,
        layer: str | None = None,
        phase: str | None = None,
        schedule: str | None = None,
        ties: str = 'first',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.payoff = payoff
        self.options = options
        self.rate = rate
        self.rule = rule
        self.intensity = intensity
        self.probability = probability
        self.phase = phase
        self.schedule = schedule
        self.ties = ties
        self.layer = layer

    def _runs_on_edges(self) -> bool:
        """Return that the laws run once for every edge of the network or lattice."""
        return True

    def _get_layer(self) -> str | None:
        """Return the layer whose edges the laws run on, or `None` for the space."""
        return self.layer

    def _check_agent_options(self) -> None:
        """Raise an error for an unknown schedule or ties, or ties not first without a schedule."""
        self._check_variant('schedule', self.schedule, (None, *ScheduleKind))
        self._check_variant('ties', self.ties, TIES)
        if self.schedule is None and self.ties != 'first':
            error_msg = (
                f'The imitation {self.label!r} has the ties {self.ties!r}, which only a rule of a schedule runs. Give '
                'a schedule, such as schedule="synchronous", or ties="first".'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Raise an error for unknown rule, schedule, or ties, or repeated options.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The rule, the schedule, or the ties are unknown, the rule `'better'` or ties other than `'first'` are
                given without a schedule, or the options are not distinct.
        """
        super()._prepare(context)
        self._check_variant('rule', self.rule, RULES, (BaseSpec,))
        if self.rule == 'better' and self.schedule is None:
            error_msg = f'The rule of imitation {self.rule!r} runs with a schedule only, such as `"synchronous"`.'
            raise ValidationError(error_msg)
        self._check_agent_options()
        if len(set(self.options)) < max(len(self.options), 2):
            error_msg = f'The options of imitation are {self.options!r}. Give at least two distinct values.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = self._get_imitation_laws(context)

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

    def _get_copies(self, member: SimpleNamespace, other: SimpleNamespace, namespace: SimpleNamespace) -> list[BaseLaw]:
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
        return [*self._get_copies(target, source, namespace), *self._get_copies(source, target, namespace)]

    @staticmethod
    def _get_draw_name(state: Expression, kind: str) -> str:
        """Return the name of a draw of a state, such as `x_activity`.

        A state read as a function of time, `x(t)`, names its draws after `x`.
        """
        name = str(state.operator()) if state.operator() is not None else str(state)
        return f'{name}_{kind}'

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
        index = categorical(self._get_draw_name(value, 'imitation'), [stay, *weights])
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
                    uniform(self._get_draw_name(value, 'imitation'))
                    < self._get_parameter_symbol('probability', namespace),
                    imitated,
                    value,
                ),
                phase=self.phase,
            ),
        ]

    def _get_imitation_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps or the update of the state by the rule of the imitation.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The jumps of the state between every pair of options, or the update of the state of every member.
        """
        if self.schedule == 'member':
            return [UpdateLaw(self._get_member(namespace, 'target'), self._imitate_member, phase=self.phase)]
        if self.schedule is not None:
            return [UpdateLaw(namespace, self._imitate, phase=self.phase)]
        if self._get_argument('probability') is not None:
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
        groups = set(system.get_groups(self._get_role('target')))
        neighbours = [other for other in agent.get_neighbors(layer=self.layer) if other.group in groups]
        given = self._get_argument('probability') is not None
        probability = self._get_parameter_value('probability', system) if given else 1.0
        if not neighbours or system.rng.random() >= probability:
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

        The agents follow the schedule, or imitate at once without one.
        """
        self._update_agents(system, self.schedule or 'synchronous', self._get_choice, self._set_choice)
