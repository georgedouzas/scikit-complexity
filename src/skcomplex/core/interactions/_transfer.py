"""Define transfers, which move amounts of states between members, the cells agents stand on, sources, and sinks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, RuleLaw, Sink, Source, TransferLaw
from ..laws.utils import where
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

FORMS = ('symbolic', 'agent')
PAIRINGS = ('members', 'random', 'pairs', 'neighbors', 'cell')
LEG_LENGTH = 3

type Amount = float | Callable[..., object]
type Leg = tuple[str | Source, str | Sink | None, Amount]


class TransferInteraction(Interaction):
    """Transfer of amounts of states from the members of a source role to the members of a target role.

    A leg moves an amount out of a state of the source member, or out of `Source()`, into a state of the target member,
    or into `Sink()`. The transfer has a first leg, `state` into `into` by `amount`, and any further `legs`, each
    `(state, into, amount)`. The amounts of all legs are read before any of them moves, so a leg sees the states as
    they were before the transfer. A guard skips the transfer where it does not hold.

    The form chooses how the transfer lowers:

    - `'symbolic'` lowers to `TransferLaw`s between the source and the target of one element, or member `i` of both
      groups. The amounts, the guard, and the rate are numbers or functions of the namespace, and the roles are read as
      `namespace.source` and `namespace.target`. With a `rate`, the amount is the size of a jump at the rate.
    - `'agent'` lowers to an agent rule, whose amounts and guard are numbers or functions of the source agent, the
      target agent, and the model view. The pairing `'members'` pairs member `i` of both groups, `'random'` pairs every
      source member with a random target member, `'pairs'` draws disjoint random pairs, `'neighbors'` pairs every
      source member with the target members in the cells or nodes around it, and `'cell'` pairs the source and target
      members that stand in one cell or node. The pairs act in random order.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        source:
            The path of the element or group the amounts leave.

        target:
            The path of the element or group the amounts enter, the source again for a transfer within members.

        state:
            The name of the state of the source the first leg takes from, or `Source()` for the environment.

        into:
            The name of the state of the target the first leg adds to, `Sink()` for the environment, or `None` for
            `state`.

        amount:
            The amount of the first leg, a number or a function.

        pairing:
            How members are paired, one of `'members'`, `'random'`, `'pairs'`, `'neighbors'`, or `'cell'`, the agent
            form alone taking the others than `'members'`.

        legs:
            Further legs, each a tuple of a state or `Source()`, a state, `Sink()`, or `None`, and an amount.

        guard:
            A function that returns whether a pair transfers, a condition of the namespace or a boolean of the agents,
            or `None` for every pair.

        rate:
            The rate of jumps of the symbolic form, a number or a function of the namespace, or `None` for a
            continuous flow.

        form:
            The lowering, `'symbolic'` to transfer laws or `'agent'` to an agent rule.

        phase:
            The phase the rule of the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The form or the pairing is unknown, a leg is malformed, or the symbolic form has an agent pairing, or the
            agent form a rate, raised by `check`.

    Examples:
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.interactions import TransferInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> from typing import Annotated
        >>> class TankElement(Element):
        ...     def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.level = level
        >>> flow = TransferInteraction(
        ...     'flow', source='upper', target='lower', state='level', amount=lambda s: s.source.level / 2
        ... )
        >>> system = System(elements=[TankElement('upper', level=4.0), TankElement('lower')], interactions=[flow])
        >>> trajectory = Simulation(system).check().prepare().run(until=1.0).trajectory_
        >>> round(float(trajectory.values('upper.level')[-1] + trajectory.values('lower.level')[-1]), 6)
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        state: str | Source,
        into: str | Sink | None = None,
        amount: Amount,
        pairing: str = 'members',
        legs: Sequence[Leg] = (),
        guard: Callable[..., object] | None = None,
        rate: Amount | None = None,
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.state = state
        self.into = into
        self.amount = amount
        self.pairing = pairing
        self.legs = legs
        self.guard = guard
        self.rate = rate
        self.form = form
        self.phase = phase

    def _get_legs(self) -> list[Leg]:
        """Return every leg with the state it adds to resolved."""
        legs: list[Leg] = []
        for state, into, amount in ((self.state, self.into, self.amount), *self.legs):
            legs.append((state, state if into is None and isinstance(state, str) else into, amount))
        return legs

    def _check_leg(self, state: object, into: object, amount: object) -> None:
        """Raise an error for a leg whose ends are not state names or the environment, or whose amount is not usable."""
        if not isinstance(state, str | Source) or not isinstance(into, str | Sink):
            error_msg = (
                f'A leg of {self.label!r} moves {state!r} into {into!r}. Give a state name or `Source()` it leaves, '
                'and a state name or `Sink()` it enters.'
            )
            raise ValidationError(error_msg)
        if isinstance(state, Source) and isinstance(into, Sink):
            error_msg = f'A leg of {self.label!r} moves from `Source()` into `Sink()`. Name a state of one end.'
            raise ValidationError(error_msg)
        if not callable(amount) and not isinstance(amount, int | float):
            error_msg = f'The amount {amount!r} of {self.label!r} is neither a number nor a function.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown form or pairing, a malformed leg, or an option the form does not take.

        Raises:
            ValidationError:
                The form or the pairing is unknown, a leg is malformed, the symbolic form has an agent pairing, or
                the agent form has a rate.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'Unknown form {self.form!r} of {self.label!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.pairing != 'members':
            error_msg = f'The symbolic form of {self.label!r} pairs members. Use `form="agent"` for {self.pairing!r}.'
            raise ValidationError(error_msg)
        if self.pairing not in PAIRINGS:
            error_msg = f'Unknown pairing {self.pairing!r} of {self.label!r}. Use one of {", ".join(PAIRINGS)}.'
            raise ValidationError(error_msg)
        for leg in self.legs:
            if not isinstance(leg, tuple) or len(leg) != LEG_LENGTH:
                error_msg = f'A leg of {self.label!r} is {leg!r}. Give a tuple `(state, into, amount)`.'
                raise ValidationError(error_msg)
        for state, into, amount in self._get_legs():
            self._check_leg(state, into, amount)
        if self.rate is not None and self.form == 'agent':
            error_msg = f'{self.label!r} has a rate, which only the symbolic form takes. Give `rate=None`.'
            raise ValidationError(error_msg)
        if self.guard is not None and not callable(self.guard):
            error_msg = f'The guard of {self.label!r} is {self.guard!r}. Give a function, or None.'
            raise ValidationError(error_msg)

    def _get_law(self, namespace: SimpleNamespace, leg: Leg) -> TransferLaw:
        """Return the transfer law of one leg between the members of a namespace."""
        state, into, amount = leg
        held = state if isinstance(state, Source) else getattr(namespace.source, state)
        sink = into if isinstance(into, Sink) else getattr(namespace.target, str(into))
        value = amount(namespace) if callable(amount) else amount
        if self.rate is None:
            if self.guard is not None:
                value = where(self.guard(namespace), value, 0)
            return TransferLaw(held, sink, value)
        rate = self.rate(namespace) if callable(self.rate) else self.rate
        if self.guard is not None:
            rate = where(self.guard(namespace), rate, 0)
        return TransferLaw(held, sink, value, rate)

    def _get_pairs(self, model: ModelViewProtocol) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of source and target agents of one step in random order."""
        sources = model.agents('source')
        targets = sources if self.source == self.target else model.agents('target')
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        if self.pairing == 'members':
            pairs = list(zip(sources, targets, strict=False))
        elif self.pairing == 'random':
            pairs = self._get_random_pairs(sources, targets, model)
        elif self.pairing == 'pairs':
            pairs = self._get_disjoint_pairs(sources, targets, model)
        else:
            pairs = self._get_located_pairs(sources, targets)
        return [pairs[int(index)] for index in model.rng.permutation(len(pairs))]

    @staticmethod
    def _get_random_pairs(
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
        model: ModelViewProtocol,
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return every source agent paired with a random target agent other than itself."""
        pairs = []
        for source in sources:
            others = [target for target in targets if target is not source]
            if others:
                pairs.append((source, others[int(model.rng.integers(len(others)))]))
        return pairs

    def _get_disjoint_pairs(
        self,
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
        model: ModelViewProtocol,
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return disjoint random pairs, of two halves of one group or of a source and a target agent."""
        first = [sources[int(index)] for index in model.rng.permutation(len(sources))]
        if self.source == self.target:
            return list(zip(first[::2], first[1::2], strict=False))
        second = [targets[int(index)] for index in model.rng.permutation(len(targets))]
        return list(zip(first, second, strict=False))

    def _get_located_pairs(
        self,
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of agents in one cell, or of every source agent with the target agents around it."""
        located: dict[object, list[AgentViewProtocol]] = {}
        for target in targets:
            located.setdefault(target.location_key, []).append(target)
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        for source in sources:
            keys = [source.location_key] if self.pairing == 'cell' else source.neighbor_cells()
            pairs.extend((source, target) for key in keys for target in located.get(key, []) if target is not source)
        return pairs

    def _evaluate(
        self,
        amount: Amount,
        source: AgentViewProtocol,
        target: AgentViewProtocol,
        model: ModelViewProtocol,
    ) -> float:
        """Return the amount of a leg between two agents as a number."""
        value = amount(source, target, model) if callable(amount) else amount
        if not isinstance(value, int | float | np.floating | np.integer):
            error_msg = f'An amount of {self.label!r} between agents is {value!r}. Return a number.'
            raise ValidationError(error_msg)
        return float(value)

    def _transfer_pairs(self, model: ModelViewProtocol) -> None:
        """Transfer between every pair of agents of one step."""
        for source, target in self._get_pairs(model):
            if source.alive and target.alive:
                self.transfer(source, target, model)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfer laws of the symbolic form, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                A transfer law per leg, or the rule that transfers between the pairs of agents.
        """
        if self.form == 'agent':
            return [RuleLaw(self._transfer_pairs, phase=self.phase)]
        return [self._get_law(namespace, leg) for leg in self._get_legs()]

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Move the amounts of every leg from one source agent to one target agent, when the guard holds.

        Args:
            source:
                The agent the amounts leave.

            target:
                The agent the amounts enter.

            model:
                The model view of the rule.

        Raises:
            ValidationError:
                An amount function returns something other than a number.
        """
        if self.guard is not None and not self.guard(source, target, model):
            return
        legs = self._get_legs()
        amounts = [self._evaluate(amount, source, target, model) for _, _, amount in legs]
        for (state, into, _), amount in zip(legs, amounts, strict=True):
            if isinstance(state, str):
                setattr(source, state, getattr(source, state) - amount)
            if isinstance(into, str):
                setattr(target, into, getattr(target, into) + amount)
