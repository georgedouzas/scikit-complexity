"""Define movements, which move the agents of a group by steps to other cells, nodes, or positions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction
from ._transfer import TransferInteraction
from ._utils import get_location_key

STEPS = ('neighbor', 'gaussian', 'levy', 'best')

type Step = str | Callable[[AgentViewProtocol, ModelViewProtocol], object]
type Cost = TransferInteraction | Callable[[AgentViewProtocol, ModelViewProtocol], None]


class MovementInteraction(Interaction):
    """Movement of the agents of a group by steps, every step of discrete time.

    A step is one of five kinds:

    - `'neighbor'` moves an agent to a uniformly drawn cell or node next to its own, on a lattice or a network.
    - `'gaussian'` adds to the position of an agent a normal vector whose components have the standard deviation
      `scale`, in a continuous space.
    - `'levy'` moves an agent in a uniformly drawn direction by a Pareto length, whose least value is `scale` and whose
      tail exponent is `exponent`, in a continuous space.
    - `'best'` moves an agent to the cell or node next to its own with the highest `score(agent, cell, model)`, ties
      broken at random.
    - A function `step(agent, model)` returns the destination, or `None` to stay.

    An agent for which `condition(agent, model)` holds attempts one step every step of time, or a Poisson number of
    steps with the mean `rate`, and takes each attempt with the probability `probability` and then with the probability
    `accept(agent, destination, model)`. An exclusive movement never moves an agent to a cell another agent of the group
    holds, and the agents move in random order. Every step taken charges its `cost`, a `TransferInteraction` from the
    agent to itself or a function of the agent and the model view. The movement is an agent rule, so the system runs in
    discrete time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group whose agents move.

        step:
            The kind of step, `'neighbor'`, `'gaussian'`, `'levy'`, or `'best'`, or a function of the agent and the
            model view that returns the destination.

        probability:
            The probability that an agent takes an attempted step.

        rate:
            The mean number of attempted steps of an agent in one unit of time, or `None` for one attempt every step.

        condition:
            A function of the agent and the model view that returns whether the agent moves, or `None` for every agent.

        accept:
            A function of the agent, its destination, and the model view that returns the probability of taking the
            step, or `None` to take every step.

        exclusive:
            Whether an agent moves only to cells or nodes no other agent of the group holds.

        score:
            The function of the agent, a neighbouring cell, and the model view that the step `'best'` maximizes.

        scale:
            The standard deviation of every component of a Gaussian step, or the least length of a Lévy step.

        exponent:
            The tail exponent of the length of a Lévy step.

        cost:
            A `TransferInteraction` that the agent pays to itself or to the environment for every step taken, a function
            of the agent and the model view run for every step taken, or `None`.

        phase:
            The phase of the step the movement runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The step is unknown, the rate is not positive, a function is not callable, or the step `'best'` has no
            score, raised by `check`.

    Examples:
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import MovementInteraction
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> walkers = GroupElement('walkers', element=Element, size=2, location=[(2, 2), (2, 2)])
        >>> walk = MovementInteraction('walk', target='walkers', step='neighbor')
        >>> space = LatticeSpace((5, 5), neighborhood='von_neumann')
        >>> system = System(elements=[walkers], interactions=[walk], space=space)
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
        >>> [bool(abs(cell - 2).sum() == 1) for cell in trajectory.agents('walkers', 1).locations]
        [True, True]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        step: Step = 'neighbor',
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        rate: float | None = None,
        condition: Callable[[AgentViewProtocol, ModelViewProtocol], bool] | None = None,
        accept: Callable[[AgentViewProtocol, object, ModelViewProtocol], float] | None = None,
        exclusive: bool = False,
        score: Callable[[AgentViewProtocol, object, ModelViewProtocol], float] | None = None,
        scale: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
        cost: Cost | None = None,
        phase: str | None = None,
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
        self.phase = phase

    def _check_functions(self) -> None:
        """Raise an error for a condition, an acceptance, a score, or a cost that cannot be called."""
        for name in ('condition', 'accept', 'score'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function, or None.'
                raise ValidationError(error_msg)
        if isinstance(self.cost, TransferInteraction):
            self.cost.check()
        elif self.cost is not None and not callable(self.cost):
            error_msg = f'The cost of {self.label!r} is {self.cost!r}. Give a TransferInteraction, a function, or None.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown step, a rate that is not positive, or a function that cannot be called.

        Raises:
            ValidationError:
                The step is unknown, the rate is not positive, a function is not callable, or the step `'best'` has
                no score.
        """
        super()._check()
        if not callable(self.step) and self.step not in STEPS:
            error_msg = f'Unknown step {self.step!r} of {self.label!r}. Use one of {", ".join(STEPS)}, or a function.'
            raise ValidationError(error_msg)
        if self.rate is not None and (isinstance(self.rate, bool) or not self.rate > 0):
            error_msg = f'The rate of {self.label!r} is {self.rate!r}. Give a positive number, or None.'
            raise ValidationError(error_msg)
        if self.step == 'best' and self.score is None:
            error_msg = f'The step `best` of {self.label!r} maximizes a score. Pass `score(agent, cell, model)`.'
            raise ValidationError(error_msg)
        self._check_functions()

    def _count_attempts(self, model: ModelViewProtocol) -> int:
        """Return the number of steps an agent attempts in one step of time."""
        return 1 if self.rate is None else int(model.rng.poisson(self.rate))

    def _get_cells(self, agent: AgentViewProtocol, occupied: set[object]) -> list[object]:
        """Return the cells or nodes around an agent that it may move to."""
        cells = agent.neighbor_cells()
        if not self.exclusive:
            return cells
        return [cell for cell in cells if get_location_key(cell) not in occupied]

    def _get_best(self, agent: AgentViewProtocol, cells: list[object], model: ModelViewProtocol) -> object:
        """Return the cell of the highest score, drawn at random among ties."""
        if self.score is None or not cells:
            return None
        scores = np.asarray([self.score(agent, cell, model) for cell in cells], dtype=float)
        best = np.flatnonzero(scores == scores.max())
        return cells[int(best[int(model.rng.integers(len(best)))])]

    def _get_destination(self, agent: AgentViewProtocol, model: ModelViewProtocol, occupied: set[object]) -> object:
        """Return the location a step of an agent leads to, or `None` to stay."""
        rng = model.rng
        if callable(self.step):
            return self.step(agent, model)
        if self.step in ('neighbor', 'best'):
            cells = self._get_cells(agent, occupied)
            if self.step == 'best':
                return self._get_best(agent, cells, model)
            return cells[int(rng.integers(len(cells)))] if cells else None
        position = np.asarray(agent.location, dtype=float)
        if self.step == 'gaussian':
            return position + rng.normal(0.0, model.parameter('scale'), position.shape)
        direction = rng.normal(size=position.shape)
        length = model.parameter('scale') * (1.0 + rng.pareto(model.parameter('exponent')))
        return position + length * direction / np.linalg.norm(direction)

    def _charge(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Charge the cost of one step taken to an agent."""
        if isinstance(self.cost, TransferInteraction):
            self.cost.transfer(agent, agent, model)
        elif self.cost is not None:
            self.cost(agent, model)

    def _take(self, agent: AgentViewProtocol, destination: object, occupied: set[object]) -> bool:
        """Move an agent to a destination unless an exclusive movement finds it held, and return whether it moved."""
        key = get_location_key(destination)
        if self.exclusive and key in occupied:
            return False
        occupied.discard(get_location_key(agent.location))
        agent.move_to(destination)
        occupied.add(key)
        return True

    def _move(
        self,
        agent: AgentViewProtocol,
        model: ModelViewProtocol,
        probability: float,
        occupied: set[object],
    ) -> None:
        """Take the attempted steps of one agent, each with its probability and acceptance, charging every step."""
        for _ in range(self._count_attempts(model)):
            if probability < 1.0 and model.rng.random() >= probability:
                continue
            destination = self._get_destination(agent, model, occupied)
            if destination is None:
                continue
            if self.accept is not None and model.rng.random() >= self.accept(agent, destination, model):
                continue
            if self._take(agent, destination, occupied):
                self._charge(agent, model)

    def _move_all(self, model: ModelViewProtocol) -> None:
        """Move every agent of the group whose condition holds, in random order."""
        agents = model.agents('target')
        occupied = {get_location_key(agent.location) for agent in agents} if self.exclusive else set()
        probability = model.parameter('probability')
        for index in model.rng.permutation(len(agents)):
            agent = agents[int(index)]
            if agent.alive and (self.condition is None or self.condition(agent, model)):
                self._move(agent, model, probability, occupied)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the movement.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that moves the agents every step.
        """
        return [RuleLaw(self._move_all, phase=self.phase)]
