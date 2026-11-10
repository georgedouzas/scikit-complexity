"""Define the interaction that moves a discrete state from one value to another."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import re
from collections.abc import Callable, Collection, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, SupportsFloat

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, exp, log

from ...errors import ValidationError
from ...specs import RoleSpec, Tags
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, JumpLaw, RuleLaw, TransitionLaw, UpdateLaw
from ..laws.utils import both, neighbors, residence, uniform, where
from ..spaces import BaseNeighborhood, BaseSpace
from ..types import AgentViewProtocol, SystemViewProtocol
from ._interaction import Interaction

KINDS = ('rate', 'probability', 'condition', 'matrix')
OVERS = ('member', 'neighbors', 'cell')
SCHEDULES = ('synchronous', 'sequential', 'random-sequential', 'member')
FORMS = ('symbolic', 'agent')


type SourceValues = float | Collection[float] | Callable[..., object]


class TransitionInteraction(Interaction):
    """An interaction that moves a discrete state from one value to another.

    A member whose state holds `source` moves to `destination`, by exactly one of these kinds:

    - `rate`: as a jump in continuous time.
    - `probability`: with that probability in each step of discrete time.
    - `condition`: in the step in which the condition holds.
    - `matrix`: by the row of a transition matrix for the current value, in discrete time, as a Markov chain. An
      `emission` matrix then draws the `observed` state from the row of the next value, as a hidden Markov model.

    With `over='neighbors'` the rate and the probability are those of one neighbour, read as `namespace.target.other`.
    The rates of the neighbours add up, and their probabilities combine as `1 - prod(1 - p)`. With `after`, a member
    moves only once it has held the source value for that many steps. In the symbolic form every function reads
    `namespace.target` and the time `namespace.t`. In the agent form every function takes the agent and the system view,
    `f(agent, system)`, and a rate `r` moves an agent with probability `1 - exp(-r)` in a step. With `over='cell'` each
    member acts on the other members of its cell, by `f(agent, other, system)`.

    Unlike `JumpInteraction`, which adds amounts to numeric states, it moves a categorical state between named values.
    Choose it when each member is in one of a few states, such as healthy, sick, and immune.

    It appears in many fields. A person moves from susceptible to infected in an epidemic on a network, a voter switches
    opinion, a cell of a forest catches fire, and a customer switches between brands.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'infection'`.

        target:
            The path of the element or group whose state moves.

        state:
            The name of the discrete state of the target.

        source:
            The value the state moves from, a collection of such values, or a function of the value of the state that
            returns whether the state moves from it, such as `lambda value: value >= 0`, a condition in the symbolic
            form. A state that moves from several values moves to the destination from each.

        destination:
            The value the state moves to.

        rate:
            The function of the rate of the move, or `None`.

        probability:
            The function of the probability of the move in a step, or `None`.

        condition:
            The function of the condition under which the state moves in a step, or `None`.

        matrix:
            The function of the transition matrix, one row and one column per value of the state, or `None`.

        emission:
            With a matrix, the function of the emission matrix, one row per value of the state and one column per value
            of the observed state, or `None`.

        observed:
            The name of the state of the target that the emission matrix draws, or `None`.

        over:
            `'member'` for a rate or a probability of the member, `'neighbors'` for that of one neighbour, or `'cell'`
            for the agent form whose members act on the members of their cell.

        neighborhood:
            The neighbourhood of `over='neighbors'`, or `None` for the neighbours of the network or the lattice.

        after:
            The number of steps a member holds the source value before it can move, or `None`.

        schedule:
            The order of the members of the agent form, `'synchronous'`, `'sequential'`, `'random-sequential'`, or
            `'member'`.

        form:
            `'symbolic'` for laws of the states of the target, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The transition has not exactly one kind, the source and the destination are equal, an emission has no
            matrix or no observed state, `over` is unknown or goes with a form or a kind it cannot take, the
            residence time is negative or goes with a rate, a matrix, or members acting on their cell, the form or the
            schedule is unknown, the symbolic form has a schedule, or a function is not callable.

    Examples:
        >>> from skcomplex.core.components.interactions import TransitionInteraction
        >>> recovery = TransitionInteraction(
        ...     'recovery', target='hosts', state='state', source=1, destination=0, rate=lambda s: 0.5
        ... )
        >>> recovery.over
        'member'
        >>> agents = TransitionInteraction(
        ...     'recovery', target='hosts', state='state', source=1, destination=2, probability=lambda a, m: 0.1,
        ...     after=3, form='agent'
        ... )
        >>> type(agents.prepare().run(SimpleNamespace()).laws_[0]).__name__
        'RuleLaw'
        >>> separation = TransitionInteraction(
        ...     'separation', target='workers', state='employer', source=lambda value: value >= 0, destination=-1,
        ...     probability=lambda a, m: 0.05, form='agent'
        ... )
        >>> separation.destination
        -1
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (JumpLaw, UpdateLaw, RuleLaw, TransitionLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the laws the arguments choose.

        These are the rules, the Markov chain steps, the jumps, or the updates.
        """
        if self._get_argument('form') == 'agent':
            return self._get_variant_tags((RuleLaw,))
        if self._get_argument('matrix') is not None:
            return self._get_variant_tags((TransitionLaw,))
        if self._get_argument('rate') is not None:
            return self._get_variant_tags((JumpLaw,))
        return self._get_variant_tags((UpdateLaw,), stochastic=self._get_argument('probability') is not None)

    def _get_draws(self) -> bool:
        """Return whether the agent form draws.

        It draws by the rows of a matrix or by a probability or a rate of moving.
        """
        chance = self._get_argument('rate') is not None or self._get_argument('probability') is not None
        drawn = self._get_argument('matrix') is not None or (self._get_argument('condition') is None and chance)
        return super()._get_draws() or (self._get_argument('form') == 'agent' and drawn)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state', 'observed'))],
        state: str = 'state',
        source: SourceValues = 0.0,
        destination: float = 1.0,
        rate: Callable[..., SupportsFloat] | None = None,
        probability: Callable[..., SupportsFloat] | None = None,
        condition: Callable[..., object] | None = None,
        matrix: Callable[..., Sequence[Sequence[object]]] | None = None,
        emission: Callable[..., Sequence[Sequence[object]]] | None = None,
        observed: str | None = None,
        over: str = 'member',
        neighborhood: BaseNeighborhood | None = None,
        after: float | None = None,
        schedule: str = 'synchronous',
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.source = source
        self.destination = destination
        self.rate = rate
        self.probability = probability
        self.condition = condition
        self.matrix = matrix
        self.emission = emission
        self.observed = observed
        self.over = over
        self.neighborhood = neighborhood
        self.after = after
        self.schedule = schedule
        self.form = form
        self.phase = phase

    def _check_kind(self) -> str:
        """Raise an error for a transition without one callable kind, or a bad emission.

        The bad emission is one it cannot take.
        """
        kinds = [kind for kind in KINDS if getattr(self, kind) is not None]
        if len(kinds) != 1 or not callable(getattr(self, kinds[0])):
            error_msg = (
                f'The transition {self.label!r} has the forms {kinds}. Give exactly one of {", ".join(KINDS)}, as a '
                'function of the namespace.'
            )
            raise ValidationError(error_msg)
        unpaired = (self.emission is None) != (self.observed is None)
        if unpaired or (self.emission is not None and self.matrix is None):
            error_msg = (
                f'The transition {self.label!r} has the emission {self.emission!r} of the state {self.observed!r}. '
                'Give both with a matrix, or neither.'
            )
            raise ValidationError(error_msg)
        return kinds[0]

    def _check_form(self) -> None:
        """Raise an error for a transition without one callable kind, or ignored arguments.

        The ignored arguments are those its kind ignores.
        """
        kind = self._check_kind()
        if self.over not in OVERS:
            error_msg = f'The transition {self.label!r} runs over {self.over!r}. Use one of {", ".join(OVERS)}.'
            raise ValidationError(error_msg)
        if self.form not in FORMS:
            error_msg = f'The form of transition {self.label!r} is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if self.schedule not in SCHEDULES or (self.form == 'symbolic' and self.schedule != 'synchronous'):
            error_msg = (
                f'The schedule of transition {self.label!r} is {self.schedule!r}. Use one of {", ".join(SCHEDULES)}, '
                'and the agent form for another than `synchronous`.'
            )
            raise ValidationError(error_msg)
        if self.over == 'cell' and (self.form != 'agent' or kind == 'matrix' or self.after is not None):
            error_msg = (
                f'The members of transition {self.label!r} act on their cell, which only the agent form does, by a '
                'rate, a probability, or a condition and without a residence time.'
            )
            raise ValidationError(error_msg)
        if self.over == 'neighbors' and (kind not in {'rate', 'probability'} or self.form == 'agent'):
            error_msg = (
                f'The transition {self.label!r} runs over neighbours, which only a rate or a probability of the '
                'symbolic form does. In the agent form, read the neighbours of the agent in the function.'
            )
            raise ValidationError(error_msg)
        single = not (callable(self.source) or isinstance(self.source, Collection))
        if self.matrix is None and single and self.source == self.destination:
            error_msg = f'The transition {self.label!r} moves from {self.source} to itself. Give distinct values.'
            raise ValidationError(error_msg)
        if self.after is not None and (kind not in {'probability', 'condition'} or not self.after >= 0):
            error_msg = (
                f'The transition {self.label!r} has the residence time {self.after!r}. Give a number of steps at least '
                '0 with a probability or a condition.'
            )
            raise ValidationError(error_msg)

    def _check_transition(self) -> None:
        """Raise an error for arguments the transition cannot take."""
        self._check_form()

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a transition without one kind, or with ignored arguments.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The transition has not exactly one kind, the source and the destination are equal, an emission has
                no matrix or no observed state, `over` is unknown or goes with a condition, a matrix, or the agent
                form, the residence time is negative or goes with a rate or a matrix, the form is unknown, or a
                function is not callable.
        """
        super()._prepare(context)
        self._check_transition()

    def _get_moving(self, value: Expression, source: SourceValues) -> Expression:
        """Return 1 for a member whose state holds a source value, and 0 otherwise."""
        if callable(source):
            return where(SR(source(value)), 1, 0)
        if isinstance(source, Collection):
            return sum((where(value == item, 1, 0) for item in set(source)), SR(0))
        return where(value == source, 1, 0)

    def _get_jump(self, value: Expression, source: SourceValues, destination: float, rate: Expression) -> JumpLaw:
        """Return the jump of the state from a source value to the destination at a rate."""
        if callable(source) or isinstance(source, Collection):
            return JumpLaw(rate * self._get_moving(value, source), {value: destination - value})
        return JumpLaw(rate * self._get_moving(value, source), {value: destination - source})

    def _get_step(self, value: Expression, source: SourceValues, destination: float, moves: Expression) -> UpdateLaw:
        """Return the update of the state that moves a member from a source value.

        It moves it where a condition holds.
        """
        if callable(source) or isinstance(source, Collection):
            return UpdateLaw(value, where(self._get_moving(value, source) > 0, where(moves, destination, value), value))
        return UpdateLaw(value, where(value == source, where(moves, destination, source), value), source=source)

    def _holds_source(self, value: float) -> bool:
        """Return whether the value of the state of an agent is a source value."""
        if callable(self.source):
            return bool(self.source(value))
        if isinstance(self.source, Collection):
            return value in set(self.source)
        return value == self.source

    def _combine_rates(self, rate: Expression) -> Expression:
        """Return the rate of the member, or the sum of the rates of its neighbours."""
        return neighbors(rate, over=self.neighborhood) if self.over == 'neighbors' else rate

    def _combine_probabilities(self, probability: Expression) -> Expression:
        """Return the probability of the member, or that at least one neighbour moves it."""
        if self.over == 'member':
            return probability
        return 1 - exp(neighbors(log(1 - probability), over=self.neighborhood))

    def _get_draw(self) -> Expression:
        """Return the uniform draw of the transition in a step."""
        return uniform(re.sub(r'\W', '_', f'transition_{self.label}'))

    def _get_chain(self, namespace: SimpleNamespace, value: Expression) -> list[BasePart]:
        """Return the transition law of the state by its matrix.

        It comes with the emission law of the observed state.
        """
        laws: list[BasePart] = []
        if self.matrix is not None:
            laws.append(TransitionLaw(value, self.matrix(namespace), label=self.label))
        if self.emission is not None and self.observed is not None:
            observed = getattr(self._get_member(namespace, 'target'), self.observed)
            laws.append(TransitionLaw(observed, self.emission(namespace), source=value, label=self.label))
        return laws

    def _get_held(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> int:
        """Return the number of whole steps an agent has held its value.

        They are counted in the storage of the system.
        """
        value = float(getattr(agent, self.state))
        record = system.storage.get(f'{self.label}.{agent.id}')
        held = record[1] + 1 if isinstance(record, tuple) and record[0] == value else 0
        system.storage[f'{self.label}.{agent.id}'] = (value, held)
        return held

    def _draw_row(self, row: Sequence[object], system: SystemViewProtocol) -> float:
        """Return a value drawn by the probabilities of a row of a matrix."""
        weights = np.asarray(row, dtype=float)
        return float(system.rng.choice(weights.size, p=weights / weights.sum()))

    def _evaluate[T](
        self,
        function: Callable[..., T],
        agent: AgentViewProtocol,
        other: AgentViewProtocol | None,
        system: SystemViewProtocol,
    ) -> T:
        """Return a function of an agent, or of an agent and a member of its cell.

        The member is the member of its cell that acts on it.
        """
        return function(agent, system) if other is None else function(agent, other, system)

    def _moves_agent(
        self,
        agent: AgentViewProtocol,
        system: SystemViewProtocol,
        other: AgentViewProtocol | None = None,
    ) -> bool:
        """Return whether an agent that holds the source value moves in this step.

        It is moved by another one or alone.
        """
        if self.condition is not None:
            return bool(self._evaluate(self.condition, agent, other, system))
        if self.rate is not None:
            probability = 1.0 - math.exp(-float(self._evaluate(self.rate, agent, other, system)))
        elif self.probability is not None:
            probability = float(self._evaluate(self.probability, agent, other, system))
        else:
            probability = 0.0
        return bool(system.rng.random() < probability)

    def _get_next(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float | None:
        """Return the next value of the state of an agent.

        It returns `None` when the agent keeps its value.
        """
        value = float(getattr(agent, self.state))
        if self.matrix is not None:
            return self._draw_row(self.matrix(agent, system)[int(value)], system)
        held = self._get_held(agent, system) if self.after is not None else 0
        if not self._holds_source(value) or (self.after is not None and held + 1 < self.after):
            return None
        return float(self.destination) if self._moves_agent(agent, system) else None

    def _get_moves(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> list[tuple[AgentViewProtocol, float]]:
        """Return the members an agent moves in this step, with their values.

        They are itself or the members of its cell.
        """
        if self.over != 'cell':
            value = self._get_next(agent, system)
            return [] if value is None else [(agent, value)]
        return [
            (other, float(self.destination))
            for other in agent.get_agents_here(agent.group)
            if other is not agent
            and self._holds_source(float(getattr(other, self.state)))
            and self._moves_agent(other, system, agent)
        ]

    def _apply(self, moves: list[tuple[AgentViewProtocol, float]], system: SystemViewProtocol) -> None:
        """Set the moved states, and draw the observed state of an emission."""
        for agent, value in moves:
            setattr(agent, self.state, value)
            if self.emission is not None and self.observed is not None:
                setattr(agent, self.observed, self._draw_row(self.emission(agent, system)[int(value)], system))

    def _move_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Move the state of one member, or of the members of its cell.

        It moves them from the states the members before it set.
        """
        if agent.alive:
            self._apply(self._get_moves(agent, system), system)

    def move(self, system: SystemViewProtocol) -> None:
        """Move the state of every agent of the target that moves in this step.

        The agents follow the schedule.

        Args:
            system:
                The system view of the interaction.
        """
        agents = list(system.get_agents(self._get_role('target')))
        if self.schedule == 'synchronous':
            moves = [move for agent in agents for move in self._get_moves(agent, system)]
            self._apply(moves, system)
            return
        order = system.rng.permutation(len(agents)) if self.schedule == 'random-sequential' else range(len(agents))
        for index in order:
            self._move_member(agents[int(index)], system)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the laws of the state, or the rule of the agent form.

        The laws are the jump, the update, the transition, or the emission law.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The law of the move, or the rule that moves the agents.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._move_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self.move, phase=self.phase)]
        value = getattr(self._get_member(namespace, 'target'), self.state)
        if self.matrix is not None:
            return self._get_chain(namespace, value)
        if self.rate is not None:
            rate = self._combine_rates(SR(self.rate(namespace)))
            return [self._get_jump(value, self.source, self.destination, rate)]
        if self.probability is not None:
            probability = self._combine_probabilities(SR(self.probability(namespace)))
            moves = self._get_draw() < probability
        else:
            moves = SR(self.condition(namespace)) if self.condition is not None else SR(0)
        if self.after is not None:
            moves = both(residence(value) + 1 >= self.after, moves)
        return [self._get_step(value, self.source, self.destination, moves)]
