"""Define life cycles, which create offspring of the agents of a group and remove agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

PLACEMENTS = ('here', 'neighbor')

type Chance = float | Callable[[AgentViewProtocol, ModelViewProtocol], float]
type Count = int | Callable[[AgentViewProtocol, ModelViewProtocol], int]
type Mutation = Callable[[dict[str, object], ModelViewProtocol], dict[str, object]]


class LifeCycleInteraction(Interaction):
    """Life cycle of the agents of a group, which give birth to offspring and die, every step of discrete time.

    Every step the agents act once each in random order. An agent dies with the probability `death`, and an agent that
    survives gives birth with the probability `birth`, or as many times as a Poisson number of the mean `rate`. A
    probability is a number or a function of the agent and the model view, and a function that returns a condition
    gives probability one where it holds. Every birth creates `offspring` new agents of the group, a number or a
    function of the parent and the model view, such as a Poisson draw. A new agent stands in the cell of its parent,
    with the placement `'here'`, or in a neighbouring cell that no agent of the group holds, with the placement
    `'neighbor'`, and a birth without such a cell creates fewer agents. A new agent copies the states `inherit` of its
    parent, the parent splits each state of `split` equally with its offspring, and `mutate(values, model)` returns the
    values of a new agent from those it would get. The other states take their defaults. The life cycle is an agent
    rule, so the system runs in discrete time.

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
            Where a new agent stands, `'here'` or `'neighbor'`.

        inherit:
            The names of the states a new agent copies from its parent.

        split:
            The names of the states the parent shares equally with its offspring.

        mutate:
            A function of the values of a new agent and the model view that returns its values, or `None`.

        phase:
            The phase of the step the life cycle runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The placement is unknown, both a birth probability and a rate are given, or a function is not callable,
            raised by `check`.

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

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        birth: Chance | None = None,
        death: Chance | None = None,
        rate: Chance | None = None,
        offspring: Count = 1,
        placement: str = 'here',
        inherit: Sequence[str] = (),
        split: Sequence[str] = (),
        mutate: Mutation | None = None,
        phase: str | None = None,
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

    def _check(self) -> None:
        """Raise an error for an unknown placement, a birth by both probability and rate, or a bad function.

        Raises:
            ValidationError:
                The placement is unknown, both a birth probability and a rate are given, or a value is neither a
                number nor a function.
        """
        super()._check()
        if self.placement not in PLACEMENTS:
            error_msg = f'Unknown placement {self.placement!r} of {self.label!r}. Use one of {", ".join(PLACEMENTS)}.'
            raise ValidationError(error_msg)
        if self.birth is not None and self.rate is not None:
            error_msg = f'{self.label!r} gives births by a probability and by a rate. Give one of them.'
            raise ValidationError(error_msg)
        for name in ('birth', 'death', 'rate', 'offspring'):
            value = getattr(self, name)
            if value is not None and not callable(value) and not isinstance(value, int | float):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a number or a function.'
                raise ValidationError(error_msg)
        if self.mutate is not None and not callable(self.mutate):
            error_msg = f'The mutation of {self.label!r} is {self.mutate!r}. Give a function, or None.'
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
        """Return the number of new agents of a parent in this step."""
        if self.rate is not None:
            births = int(model.rng.poisson(self._evaluate(self.rate, agent, model)))
        else:
            births = int(self._happens(self.birth, agent, model))
        return births * int(self._evaluate(self.offspring, agent, model)) if births else 0

    def _get_places(
        self,
        agent: AgentViewProtocol,
        count: int,
        occupied: set[object],
        model: ModelViewProtocol,
    ) -> list[object]:
        """Return the locations of the new agents of a parent, fewer when free neighbouring cells run out."""
        if self.placement == 'here':
            return [agent.location] * count
        free = [cell for cell in agent.neighbor_cells() if cell not in occupied]
        chosen = [free[int(index)] for index in model.rng.permutation(len(free))[:count]]
        occupied.update(chosen)
        return chosen

    def _bear(self, agent: AgentViewProtocol, places: list[object], model: ModelViewProtocol) -> None:
        """Create the new agents of a parent at their places, with inherited, split, and mutated values."""
        shares = {name: getattr(agent, name) / (len(places) + 1) for name in self.split}
        for name, share in shares.items():
            setattr(agent, name, share)
        for place in places:
            values: dict[str, object] = {name: getattr(agent, name) for name in self.inherit}
            values.update(shares)
            if self.mutate is not None:
                values = self.mutate(values, model)
            agent.create(None, place, **values)

    def _live(self, model: ModelViewProtocol) -> None:
        """Remove the agents that die and create the offspring of those that give birth, in random order."""
        agents = model.agents('target')
        occupied = {agent.location_key for agent in agents} if self.placement == 'neighbor' else set()
        for index in model.rng.permutation(len(agents)):
            agent = agents[int(index)]
            if not agent.alive:
                continue
            if self._happens(self.death, agent, model):
                occupied.discard(agent.location_key)
                agent.remove()
                continue
            count = self._count_births(agent, model)
            if count:
                self._bear(agent, self._get_places(agent, count, occupied, model), model)

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
