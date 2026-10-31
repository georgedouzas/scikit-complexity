"""Define random walks, which move the members of a group by random steps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

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

STEPS = ('neighbor', 'gaussian', 'levy')


class RandomWalkInteraction(Interaction):
    """Random walk of the members of a group, which take random steps every step of discrete time.

    A step is one of three kinds:

    - `'neighbor'` moves a member to a uniformly drawn cell or node next to its own, on a lattice or a network.
    - `'gaussian'` adds to the position of a member a normal vector whose components have the standard deviation
      `scale`, in a continuous space.
    - `'levy'` moves a member in a uniformly drawn direction by a Pareto length, whose least value is `scale` and whose
      tail exponent is `exponent`, in a continuous space.

    A member attempts one step every step of time, or a Poisson number of steps with the mean `rate`, and takes each
    attempt with the probability `probability`. Every step taken lowers the state `budget` of the member by `cost`.
    The walk is a rule, so the system runs in discrete time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        walkers:
            The path of the group whose members walk.

        step:
            The kind of step, `'neighbor'`, `'gaussian'`, or `'levy'`.

        probability:
            The probability that a member takes an attempted step.

        scale:
            The standard deviation of every component of a Gaussian step, or the least length of a Lévy step.

        exponent:
            The tail exponent of the length of a Lévy step.

        rate:
            The mean number of attempted steps of a member in one unit of time, or `None` for one attempt every step.

        cost:
            The amount a step taken subtracts from the state `budget` of the member.

        budget:
            The name of the state of the walkers the cost is charged to, or `None` for walks without a cost.

        phase:
            The phase of the step the walk runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import RandomWalkInteraction
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> walkers = GroupElement('walkers', element=Element, size=2, location=[(2, 2), (2, 2)])
        >>> walk = RandomWalkInteraction('walk', walkers='walkers', step='neighbor')
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
        walkers: Annotated[str, RoleSpec()],
        step: str = 'neighbor',
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        scale: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
        rate: float | None = None,
        cost: Annotated[float, ParameterSpec()] = 0.0,
        budget: str | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.walkers = walkers
        self.step = step
        self.probability = probability
        self.scale = scale
        self.exponent = exponent
        self.rate = rate
        self.cost = cost
        self.budget = budget
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for an unknown step, a rate that is not positive, or a budget that is not a state name.

        Raises:
            ValidationError:
                The step is unknown, the rate is not positive, or the budget is not a string.
        """
        super()._check()
        if self.step not in STEPS:
            error_msg = f'Unknown step {self.step!r} of {self.label!r}. Use one of {", ".join(STEPS)}.'
            raise ValidationError(error_msg)
        if self.rate is not None and (isinstance(self.rate, bool) or not self.rate > 0):
            error_msg = f'The rate of {self.label!r} is {self.rate!r}. Give a positive number, or None.'
            raise ValidationError(error_msg)
        if self.budget is not None and not isinstance(self.budget, str):
            error_msg = f'The budget of {self.label!r} is {self.budget!r}. Give the name of a state, or None.'
            raise ValidationError(error_msg)

    def _count_attempts(self, model: ModelViewProtocol) -> int:
        """Return the number of steps a member attempts in one step of time."""
        return 1 if self.rate is None else int(model.rng.poisson(self.rate))

    def _get_destination(self, walker: AgentViewProtocol, model: ModelViewProtocol) -> object:
        """Return the location a step of a walker leads to."""
        rng = model.rng
        if self.step == 'neighbor':
            cells = walker.neighbor_cells()
            return cells[int(rng.integers(len(cells)))] if cells else walker.location
        position = np.asarray(walker.location, dtype=float)
        if self.step == 'gaussian':
            return position + rng.normal(0.0, model.parameter('scale'), position.shape)
        direction = rng.normal(size=position.shape)
        length = model.parameter('scale') * (1.0 + rng.pareto(model.parameter('exponent')))
        return position + length * direction / np.linalg.norm(direction)

    def _move(self, walker: AgentViewProtocol, model: ModelViewProtocol, probability: float, cost: float) -> None:
        """Take the attempted steps of one walker, each with its probability, and charge the cost of every step."""
        for _ in range(self._count_attempts(model)):
            if probability < 1.0 and model.rng.random() >= probability:
                continue
            walker.move_to(self._get_destination(walker, model))
            if self.budget is not None:
                setattr(walker, self.budget, getattr(walker, self.budget) - cost)

    def _walk(self, model: ModelViewProtocol) -> None:
        """Move every walker of the group by its steps of one step of time."""
        probability, cost = model.parameter('probability'), model.parameter('cost')
        for walker in model.agents('walkers'):
            if walker.alive:
                self._move(walker, model, probability, cost)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the walk.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that moves the walkers every step.
        """
        return [RuleLaw(self._walk, phase=self.phase)]
