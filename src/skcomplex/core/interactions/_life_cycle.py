"""Define life cycles, which create offspring of the agents of a group and remove agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

PLACEMENTS = ('here', 'neighbor')

type Chance = float | Callable[..., float]
type Count = int | Callable[..., int]
type Mutation = Callable[[dict[str, object], ModelViewProtocol], dict[str, object]]
type Arrival = float | Callable[..., float]
type Values = Callable[..., Mapping[str, object]]
type Placement = str | Callable[..., object]
type Kinship = Callable[..., object]


class LifeCycleInteraction(Interaction):
    """Life cycle of the agents of a group, which give birth to offspring and die, every step of discrete time.

    Every step the agents act once each in random order. An agent dies with the probability `death`, and an agent that
    survives gives birth with the probability `birth`, or as many times as a Poisson number of the mean `rate`. A
    probability is a number or a function of the agent and the model view, and a function that returns a condition
    gives probability one where it holds. Every birth creates `offspring` new agents, a number or a function of the
    parent and the model view, such as a Poisson draw, of the group `group`, or of the parent's group without one. A
    new agent stands in the cell of its parent, with the placement `'here'`, in a neighbouring cell that no agent of
    the group holds, with the placement `'neighbor'`, or at the location a function of the parent and the model view
    returns, where `None` means no place, and a birth without a place creates fewer agents.

    A new agent copies the states `inherit` of its parent and takes the values that `values(parent, model)` returns,
    such as a generation one higher or drawn states. The parent splits each state of `split` with its offspring,
    keeping the share `keep`, a number or a function of the parent and the model view such as a random cut, or else an
    equal share, and the offspring share the rest equally. Then `mutate(values, model)` returns the values of a new
    agent from those it would get, and the other states take their defaults. A parent that gave birth takes the values
    `reset`, and with `replace` it is removed, so its offspring replace it, also when the birth created none. With
    `renew` an agent that dies is renewed in place instead of removed: it keeps its identity, its place, and the states
    the new values leave out, and takes the values of an offspring of its own. With the probability `cull` an agent
    that survives removes one member of `group` for which `kin(agent, member)` holds, the first in the order of the
    group, such as one it owns. Every step a Poisson number of the mean `arrival`, a number or a function of the model
    view, of new agents arrive without a parent, at the location of the placement function or else at a random place,
    with the values `values(None, model)`. The life cycle reports the `births`, `deaths`, and `arrivals` of every step.
    It is an agent rule, so the system runs in discrete time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the variable group whose agents give birth and die.

        birth:
            The probability that an agent gives birth in a step, or `None` for no births by probability.

        death:
            The probability that an agent dies in a step, or `None` for no deaths.

        rate:
            The mean number of births of an agent in a step, a number or a function, or `None` for no births by rate.

        offspring:
            The number of new agents of a birth.

        placement:
            Where a new agent stands, `'here'`, `'neighbor'`, or a function of the parent, `None` for an arrival, and
            the model view that returns its location.

        inherit:
            The names of the states a new agent copies from its parent.

        split:
            The names of the states the parent shares with its offspring.

        mutate:
            A function of the values of a new agent and the model view that returns its values, or `None`.

        phase:
            The phase of the step the life cycle runs in, or `None` for the default phase.

        group:
            The path of the variable group the new agents join, or `None` for the group of the target.

        values:
            A function of the parent, `None` for an arrival, and the model view that returns values of a new agent by
            state name, or `None`.

        keep:
            The share of every split state the parent keeps, a number or a function, or `None` for an equal share.

        reset:
            The values a parent takes after it gives birth, by state name, or `None`.

        replace:
            Whether a parent that gives birth is removed.

        renew:
            Whether an agent that dies is renewed in place instead of removed.

        cull:
            The probability that an agent removes one member of `group` in a step, or `None` for no removals.

        kin:
            A function of an agent and a member of `group` that returns whether the agent may remove the member, or
            `None` for every member.

        arrival:
            The mean number of new agents that arrive in a step without a parent, or `None` for no arrivals.

    Raises:
        ValidationError:
            The placement is unknown, both a birth probability and a rate are given, a function is not callable, or a
            removal of members of another group has no group, raised by `check`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import LifeCycleInteraction
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, visual=None, *, size: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.size = size
        >>> cells = GroupElement('cells', element=CellElement, size=1, values={'size': 8.0}, variable=True)
        >>> division = LifeCycleInteraction('division', target='cells', birth=1.0, split=('size',))
        >>> system = System(elements=[cells], interactions=[division], space=LatticeSpace((3, 3)))
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=2).trajectory_
        >>> trajectory.agents('cells', 2).values['size'].tolist()
        [2.0, 2.0, 2.0, 2.0]
    """

    reports: ClassVar[tuple[str, ...]] = ('births', 'deaths', 'arrivals')

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        birth: Chance | None = None,
        death: Chance | None = None,
        rate: Chance | None = None,
        offspring: Count = 1,
        placement: Placement = 'here',
        inherit: Sequence[str] = (),
        split: Sequence[str] = (),
        mutate: Mutation | None = None,
        phase: str | None = None,
        group: Annotated[str | None, RoleSpec(optional=True)] = None,
        values: Values | None = None,
        keep: Chance | None = None,
        reset: Mapping[str, object] | None = None,
        replace: bool = False,
        renew: bool = False,
        cull: Chance | None = None,
        kin: Kinship | None = None,
        arrival: Arrival | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.birth = birth
        self.death = death
        self.rate = rate
        self.offspring = offspring
        self.placement = placement
        self.inherit = inherit
        self.split = split
        self.mutate = mutate
        self.phase = phase
        self.group = group
        self.values = values
        self.keep = keep
        self.reset = reset
        self.replace = replace
        self.renew = renew
        self.cull = cull
        self.kin = kin
        self.arrival = arrival

    def _check(self) -> None:
        """Raise an error for an unknown placement, a birth by both probability and rate, or a bad function.

        Raises:
            ValidationError:
                The placement is unknown, both a birth probability and a rate are given, a value is neither a
                number nor a function, or a removal of members of another group has no group.
        """
        super()._check()
        if not callable(self.placement) and self.placement not in PLACEMENTS:
            error_msg = (
                f'Unknown placement {self.placement!r} of {self.label!r}. Use one of {", ".join(PLACEMENTS)}, or a '
                'function.'
            )
            raise ValidationError(error_msg)
        if self.birth is not None and self.rate is not None:
            error_msg = f'{self.label!r} gives births by a probability and by a rate. Give one of them.'
            raise ValidationError(error_msg)
        for name in ('birth', 'death', 'rate', 'offspring', 'keep', 'cull', 'arrival'):
            value = getattr(self, name)
            if value is not None and not callable(value) and not isinstance(value, int | float):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a number or a function.'
                raise ValidationError(error_msg)
        for name in ('mutate', 'values', 'kin'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function, or None.'
                raise ValidationError(error_msg)
        if self.cull is not None and self.group is None:
            error_msg = f'{self.label!r} removes members of another group but binds no group. Give the group.'
            raise ValidationError(error_msg)

    @staticmethod
    def _evaluate(value: Chance | Count, agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
        """Return a number, or the value of a function of an agent and the model view, as a float."""
        return float(value(agent, model) if callable(value) else value)

    def _happens(self, chance: Chance | None, agent: AgentViewProtocol, model: ModelViewProtocol) -> bool:
        """Return whether an event of a probability happens to an agent in this step."""
        if chance is None:
            return False
        probability = self._evaluate(chance, agent, model)
        return probability >= 1.0 or (probability > 0.0 and model.rng.random() < probability)

    def _count_births(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> int:
        """Return the number of births of a parent in this step."""
        if self.rate is not None:
            return int(model.rng.poisson(self._evaluate(self.rate, agent, model)))
        return int(self._happens(self.birth, agent, model))

    def _get_places(
        self,
        agent: AgentViewProtocol,
        count: int,
        occupied: set[object],
        model: ModelViewProtocol,
    ) -> list[object]:
        """Return the locations of the new agents of a parent, fewer when places run out."""
        if callable(self.placement):
            places = [self.placement(agent, model) for _ in range(count)]
            return [place for place in places if place is not None]
        if self.placement == 'here':
            return [agent.location] * count
        free = [cell for cell in agent.neighbor_cells() if cell not in occupied]
        chosen = [free[int(index)] for index in model.rng.permutation(len(free))[:count]]
        occupied.update(chosen)
        return chosen

    def _get_values(self, agent: AgentViewProtocol | None, model: ModelViewProtocol) -> dict[str, object]:
        """Return the inherited and given values of a new agent of a parent, or of an arrival."""
        values: dict[str, object] = {} if agent is None else {name: getattr(agent, name) for name in self.inherit}
        if self.values is not None:
            values.update(self.values(agent, model))
        return values

    def _create(
        self,
        agent: AgentViewProtocol | None,
        place: object,
        values: dict[str, object],
        model: ModelViewProtocol,
    ) -> None:
        """Create a new agent with mutated values at a place, `None` for a random one, in the group of the cycle."""
        if self.mutate is not None:
            values = self.mutate(values, model)
        if self.group is not None:
            model.create('group', place, **values)
        elif agent is not None:
            agent.create(None, place, **values)
        else:
            model.create('target', place, **values)

    def _bear(self, agent: AgentViewProtocol, places: list[object], model: ModelViewProtocol) -> None:
        """Create the new agents of a parent at their places, with inherited, given, split, and mutated values."""
        kept = 1 / (len(places) + 1) if self.keep is None else self._evaluate(self.keep, agent, model)
        totals = {name: float(getattr(agent, name)) for name in self.split}
        for name, amount in totals.items():
            setattr(agent, name, amount * kept)
        shares = {name: (amount - amount * kept) / max(len(places), 1) for name, amount in totals.items()}
        for place in places:
            values = self._get_values(agent, model)
            values.update(shares)
            self._create(agent, place, values, model)

    def _renew(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Give an agent that dies the values of an offspring of its own, in place."""
        values = self._get_values(agent, model)
        if self.mutate is not None:
            values = self.mutate(values, model)
        for name, value in values.items():
            setattr(agent, name, value)

    def _remove_kin(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Remove the first member of the group of the cycle that is kin of an agent, if there is one."""
        kin = self.kin
        members = (item for item in model.agents('group') if item.alive and (kin is None or kin(agent, item)))
        member = next(members, None)
        if member is not None:
            member.remove()

    def _die(self, agent: AgentViewProtocol, occupied: set[object], model: ModelViewProtocol) -> None:
        """Renew an agent in place, or remove it and free its cell."""
        if self.renew:
            self._renew(agent, model)
            return
        occupied.discard(agent.location_key)
        agent.remove()

    def _act(self, agent: AgentViewProtocol, occupied: set[object], model: ModelViewProtocol) -> tuple[int, int]:
        """Let an agent die, or give birth and remove kin, and return its numbers of new agents and deaths."""
        if self._happens(self.death, agent, model):
            self._die(agent, occupied, model)
            return 0, 1
        born = 0
        events = self._count_births(agent, model)
        if events > 0:
            births = events * int(self._evaluate(self.offspring, agent, model))
            places = self._get_places(agent, births, occupied, model) if births else []
            if places or self.split:
                self._bear(agent, places, model)
            born = len(places)
            for name, value in (self.reset or {}).items():
                setattr(agent, name, value)
            if self.replace:
                occupied.discard(agent.location_key)
                agent.remove()
                return born, 1
        if self._happens(self.cull, agent, model):
            self._remove_kin(agent, model)
        return born, 0

    def _arrive(self, model: ModelViewProtocol) -> int:
        """Create the agents that arrive without a parent in this step, and return their number."""
        if self.arrival is None:
            return 0
        mean = self.arrival(model) if callable(self.arrival) else self.arrival
        count = int(model.rng.poisson(float(mean)))
        for _ in range(count):
            place = self.placement(None, model) if callable(self.placement) else None
            self._create(None, place, self._get_values(None, model), model)
        return count

    def _live(self, model: ModelViewProtocol) -> None:
        """Remove or renew the agents that die and create the offspring of those that give birth, in random order."""
        agents = model.agents('target')
        occupied = {agent.location_key for agent in agents} if self.placement == 'neighbor' else set()
        births = deaths = 0
        for index in model.rng.permutation(len(agents)):
            agent = agents[int(index)]
            if not agent.alive:
                continue
            born, died = self._act(agent, occupied, model)
            births, deaths = births + born, deaths + died
        model.report('births', births)
        model.report('deaths', deaths)
        model.report('arrivals', self._arrive(model))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the life cycle.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that removes and creates agents every step.
        """
        return [RuleLaw(self._live, phase=self.phase)]
