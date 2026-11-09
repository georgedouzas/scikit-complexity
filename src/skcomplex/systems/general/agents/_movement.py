"""Define movements.

A movement moves the agents of a group by steps to other cells, nodes, or positions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction, TransferInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import ElementKind

STEPS = ('neighbor', 'gaussian', 'levy', 'best')
SCHEDULES = ('sequential', 'random-sequential', 'member')
TIES = ('random', 'first', 'nearest')
GRID_STEPS = ('neighbor', 'best')

type Step = str | Callable[..., object]
type Cost = TransferInteraction | Callable[[AgentViewProtocol, SystemViewProtocol], None]
type Vision = float | Callable[[AgentViewProtocol, SystemViewProtocol], float]


class MovementInteraction(Interaction):
    """Movement of the agents of a group by steps, every step of discrete time.

    A step is one of five kinds:

    - `'neighbor'` moves an agent to a uniformly drawn cell or node next to its own, on a lattice or a network.
    - `'gaussian'` adds to the position of an agent a normal vector whose components have the standard deviation
      `scale`, in a continuous space.
    - `'levy'` moves an agent in a uniformly drawn direction by a Pareto length, whose least value is `scale` and whose
      tail exponent is `exponent`, in a continuous space.
    - `'best'` moves an agent to the cell or node next to its own with the highest `score(agent, cell, system)`. With a
      `vision`, a number or a function of the agent and the system view, it looks instead along every direction of the
      neighbourhood of a lattice up to that many cells, and at the cell of the agent, in the order of the axes, the
      positive direction first. Among cells of equal highest score, the ties `'random'` draw one uniformly, `'first'`
      take the first in that order, and `'nearest'` take the nearest, the first of them among equal distances. By
      default the ties are `'random'` without a vision and `'nearest'` with one.
    - A function `step(agent, system)` returns the destination, or `None` to stay.

    An agent for which `condition(agent, system)` holds attempts one step every step of time, or a Poisson number of
    steps with the mean `rate`, and takes each attempt with the probability `probability` and then with the probability
    `accept(agent, destination, system)`. An exclusive movement never moves an agent to a cell another agent of the
    group holds. The agents move one after the other, in their order under the schedule `'sequential'`, in a new random
    order every step under `'random-sequential'`, or each in its own activation under `'member'`, together with the
    rules of its element and the other member interactions, in the order the system lists them. Every step taken charges
    its `cost`, a `TransferInteraction` from the agent to itself or a function of the agent and the system view. The
    movement is an agent rule, so the system runs in discrete time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group whose agents move.

        step:
            The kind of step, `'neighbor'`, `'gaussian'`, `'levy'`, or `'best'`, or a function of the agent and the
            system view that returns the destination.

        probability:
            The probability that an agent takes an attempted step.

        rate:
            The mean number of attempted steps of an agent in one unit of time, or `None` for one attempt every step.

        condition:
            A function of the agent and the system view that returns whether the agent moves, or `None` for every agent.

        accept:
            A function of the agent, its destination, and the system view that returns the probability of taking the
            step, or `None` to take every step.

        exclusive:
            Whether an agent moves only to cells or nodes no other agent of the group holds.

        score:
            The function of the agent, a neighbouring cell, and the system view that the step `'best'` maximizes.

        scale:
            The standard deviation of every component of a Gaussian step, or the least length of a Lévy step.

        exponent:
            The tail exponent of the length of a Lévy step.

        cost:
            A `TransferInteraction` that the agent pays to itself or to the environment for every step taken, a function
            of the agent and the system view run for every step taken, or `None`.

        vision:
            The number of cells the step `'best'` looks along every direction of a lattice, a number or a function of
            the agent and the system view, or `None` for the cells next to the agent.

        schedule:
            The order of the agents, `'sequential'`, `'random-sequential'`, or `'member'`.

        phase:
            The phase of the step the movement runs in, or `None` for the default phase.

        ties:
            The cell the step `'best'` takes among cells of equal highest score, `'random'`, `'first'`, or `'nearest'`,
            or `None` for `'random'` without a vision and `'nearest'` with one.

    Raises:
        ValidationError:
            The step, the schedule, or the ties are unknown, the rate is not positive, a function is not callable, or
            the step `'best'` has no score, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.systems.general.agents import MovementInteraction
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> walkers = GroupElement('walkers', element=Element, size=2, location=[(2, 2), (2, 2)])
        >>> walk = MovementInteraction('walk', target='walkers', step='neighbor')
        >>> space = LatticeSpace((5, 5), metric='manhattan')
        >>> system = CompositeSystem(elements=[walkers], interactions=[walk], space=space)
        >>> trajectory = simulate(system, until=1, random_state=0)
        >>> [bool(abs(cell - 2).sum() == 1) for cell in trajectory.agents('walkers', 1).locations]
        [True, True]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    def _draws(self) -> bool:
        """Return whether the movement draws.

        It draws by its order, a random step, a probability or rate, or random ties.
        """
        ties = self.ties or ('random' if self.vision is None else 'nearest')
        drawn = self.step in ('neighbor', 'gaussian', 'levy') or (self.step == 'best' and ties == 'random')
        return super()._draws() or drawn or self.probability != 1 or self.rate is not None or self.accept is not None

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP))],
        step: Step = 'neighbor',
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        rate: float | None = None,
        condition: Callable[[AgentViewProtocol, SystemViewProtocol], bool] | None = None,
        accept: Callable[[AgentViewProtocol, object, SystemViewProtocol], float] | None = None,
        exclusive: bool = False,
        score: Callable[[AgentViewProtocol, object, SystemViewProtocol], float] | None = None,
        scale: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
        cost: Cost | None = None,
        vision: Vision | None = None,
        schedule: str = 'random-sequential',
        phase: str | None = None,
        ties: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.step = step
        self.probability = probability
        self.rate = rate
        self.condition = condition
        self.accept = accept
        self.exclusive = exclusive
        self.score = score
        self.scale = scale
        self.exponent = exponent
        self.cost = cost
        self.vision = vision
        self.schedule = schedule
        self.phase = phase
        self.ties = ties

    def _check_functions(self) -> None:
        """Raise an error for a condition, acceptance, score, or cost that is not callable."""
        for name in ('condition', 'accept', 'score'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function, or None.'
                raise ValidationError(error_msg)
        if isinstance(self.cost, TransferInteraction):
            self.cost.prepare()
        elif self.cost is not None and not callable(self.cost):
            error_msg = f'The cost of {self.label!r} is {self.cost!r}. Give a TransferInteraction, a function, or None.'
            raise ValidationError(error_msg)

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for a step to a neighbouring cell or node in a continuous space."""
        if isinstance(self.step, str) and self.step in GRID_STEPS:
            self._check_grid('step', self.step, space, "step='gaussian' or step='levy', or a step function")

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown step, a rate not positive, or a bad function.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The step, the schedule, or the ties are unknown, the rate is not positive, a function is not callable,
                or the step `'best'` has no score.
        """
        super()._prepare(context)
        self._check_variant('step', self.step, STEPS, functions=True)
        if self.rate is not None and (isinstance(self.rate, bool) or not self.rate > 0):
            error_msg = f'The rate of {self.label!r} is {self.rate!r}. Give a positive number, or None.'
            raise ValidationError(error_msg)
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r} of {self.label!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValidationError(error_msg)
        if self.ties is not None:
            self._check_variant('ties', self.ties, TIES)
        if self.vision is not None and not callable(self.vision) and not isinstance(self.vision, int | float):
            error_msg = f'The vision of {self.label!r} is {self.vision!r}. Give a number, a function, or None.'
            raise ValidationError(error_msg)
        if self.step == 'best' and self.score is None:
            error_msg = f'The step `best` of {self.label!r} maximizes a score. Pass `score(agent, cell, system)`.'
            raise ValidationError(error_msg)
        self._check_functions()

    def _count_attempts(self, system: SystemViewProtocol) -> int:
        """Return the number of steps an agent attempts in one step of time."""
        return 1 if self.rate is None else int(system.rng.poisson(self.rate))

    def _get_cells(self, agent: AgentViewProtocol, occupied: set[object]) -> list[object]:
        """Return the cells or nodes around an agent that it may move to."""
        cells = agent.neighbor_cells()
        if not self.exclusive:
            return cells
        return [cell for cell in cells if cell not in occupied]

    def _get_best(
        self,
        agent: AgentViewProtocol,
        cells: list[object],
        system: SystemViewProtocol,
        distances: list[int] | None = None,
    ) -> object:
        """Return the cell of the highest score, chosen among ties by the tie rule."""
        if self.score is None or not cells:
            return None
        scores = [self.score(agent, cell, system) for cell in cells]
        ties = self.ties or ('random' if distances is None else 'nearest')
        if ties != 'nearest':
            return cells[self._pick_best(scores, ties, system)]
        values = np.asarray(scores, dtype=float)
        best = np.flatnonzero(values == values.max())
        near = np.asarray([1 if distances is None else distances[int(index)] for index in best])
        return cells[int(best[near == near.min()][0])]

    @staticmethod
    def _shift(here: tuple[int, ...], cell: tuple[int, ...], system: SystemViewProtocol) -> tuple[int, ...]:
        """Return the unit step from a cell to a neighbouring cell.

        The step goes across the wrapped edge of a torus.
        """
        shift = [b - a for a, b in zip(here, cell, strict=True)]
        if system.torus:
            sizes = [int(item) for item in system.dimensions or ()]
            shift = [d - round(d / n) * n for d, n in zip(shift, sizes, strict=True)]
        return tuple(shift)

    def _get_sight(
        self,
        agent: AgentViewProtocol,
        occupied: set[object],
        system: SystemViewProtocol,
    ) -> tuple[list[object], list[int]]:
        """Return the cells in sight of an agent along every direction of a lattice.

        Its own cell comes first, with the distances.
        """
        vision = self.vision(agent, system) if callable(self.vision) else self.vision
        reach = int(vision or 0)
        here = tuple(int(item) for item in np.atleast_1d(np.asarray(agent.location_key)))
        sizes = [int(item) for item in system.dimensions or ()]
        cells: list[object] = [agent.location_key]
        distances = [0]
        shifts = [
            self._shift(here, tuple(int(item) for item in np.atleast_1d(np.asarray(neighbor))), system)
            for neighbor in agent.neighbor_cells()
        ]
        for shift in sorted(shifts, key=lambda step: tuple((item == 0, -item) for item in step)):
            for distance in range(1, reach + 1):
                point = [a + distance * d for a, d in zip(here, shift, strict=True)]
                if system.torus:
                    point = [item % size for item, size in zip(point, sizes, strict=True)]
                elif any(not 0 <= item < size for item, size in zip(point, sizes, strict=True)):
                    break
                cell = tuple(point)
                if not self.exclusive or cell not in occupied:
                    cells.append(cell)
                    distances.append(distance)
        return cells, distances

    def _get_destination(self, agent: AgentViewProtocol, system: SystemViewProtocol, occupied: set[object]) -> object:
        """Return the location a step of an agent leads to, or `None` to stay."""
        rng = system.rng
        if callable(self.step):
            return self.step(agent, system)
        if self.step == 'best' and self.vision is not None:
            cells, distances = self._get_sight(agent, occupied, system)
            return self._get_best(agent, cells, system, distances)
        if self.step in ('neighbor', 'best'):
            cells = self._get_cells(agent, occupied)
            if self.step == 'best':
                return self._get_best(agent, cells, system)
            return cells[int(rng.integers(len(cells)))] if cells else None
        position = np.asarray(agent.location, dtype=float)
        if self.step == 'gaussian':
            return position + rng.normal(0.0, self._get_parameter_value('scale', system), position.shape)
        direction = rng.normal(size=position.shape)
        length = self._get_parameter_value('scale', system) * (
            1.0 + rng.pareto(self._get_parameter_value('exponent', system))
        )
        return position + length * direction / np.linalg.norm(direction)

    def _charge(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Charge the cost of one step taken to an agent."""
        if isinstance(self.cost, TransferInteraction):
            self.cost.transfer(agent, agent, system)
        elif self.cost is not None:
            self.cost(agent, system)

    def _take(self, agent: AgentViewProtocol, destination: object, occupied: set[object]) -> bool:
        """Move an agent to a destination unless an exclusive movement finds it held.

        It returns whether it moved.
        """
        if not self.exclusive:
            agent.move_to(destination)
            return True
        if destination == agent.location_key:
            return True
        if destination in occupied:
            return False
        occupied.discard(agent.location_key)
        agent.move_to(destination)
        occupied.add(agent.location_key)
        return True

    def _move(
        self,
        agent: AgentViewProtocol,
        system: SystemViewProtocol,
        probability: float,
        occupied: set[object],
    ) -> None:
        """Take the attempted steps of one agent.

        Each step has its probability and acceptance, and every step is charged.
        """
        for _ in range(self._count_attempts(system)):
            if probability < 1.0 and system.rng.random() >= probability:
                continue
            destination = self._get_destination(agent, system, occupied)
            if destination is None:
                continue
            if self.accept is not None and system.rng.random() >= self.accept(agent, destination, system):
                continue
            if self._take(agent, destination, occupied):
                self._charge(agent, system)

    def _move_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Move one agent whose condition holds, in its own activation."""
        if agent.alive and (self.condition is None or self.condition(agent, system)):
            occupied = (
                {other.location_key for other in system.agents(self._get_role('target'))} if self.exclusive else set()
            )
            self._move(agent, system, self._get_parameter_value('probability', system), occupied)

    def _move_all(self, system: SystemViewProtocol) -> None:
        """Move every agent of the group whose condition holds.

        They move in their order or in random order.
        """
        agents = system.agents(self._get_role('target'))
        occupied = {agent.location_key for agent in agents} if self.exclusive else set()
        probability = self._get_parameter_value('probability', system)
        order = system.rng.permutation(len(agents)) if self.schedule == 'random-sequential' else range(len(agents))
        for index in order:
            agent = agents[int(index)]
            if agent.alive and (self.condition is None or self.condition(agent, system)):
                self._move(agent, system, probability, occupied)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the movement.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that moves the agents every step.
        """
        if self.schedule == 'member':
            return [RuleLaw(self._move_member, phase=self.phase, members=self._get_role('target'))]
        return [RuleLaw(self._move_all, phase=self.phase)]
