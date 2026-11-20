"""Queue systems: tasks executed by priority, whose waiting times are heavy tailed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class TaskView(AgentViewProtocol, Protocol):
    """View of a task of the priority queue system.

    Its states and parameters are attributes.
    """

    priority: float
    age: float
    wait: float
    greed: float


class TaskElement(Element):
    """Task of the Barabási priority queue.

    It holds a priority, an age, and the waiting time when it was last executed.

    Args:
        label:
            The label of the agent or group.

        priority:
            The priority, uniform on 0 to 1 for a new task.

        age:
            The number of steps the task has waited.

        wait:
            The waiting time of the last task executed from this slot of the list.

        greed:
            The probability that the highest priority task is executed rather than a random one.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.organizations import TaskElement
        >>> group = TaskElement('tasks', priority=[0.2, 0.7], size=2)
        >>> group.label, type(group).__name__, group.size
        ('tasks', 'TaskElement', 2)
    """

    visual = VisualSpec('histograms', {'color_by': 'age', 'shown': ('agents',)})

    def __init__(
        self,
        label: str,
        *,
        priority: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        age: Annotated[object, NN] = 0.0,
        wait: Annotated[object, NN] = 0.0,
        greed: Annotated[object, RealSet.closed_open(0, oo)] = 0.99999,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.priority = priority
        self.age = age
        self.wait = wait
        self.greed = greed


class ExecutionInteraction(Interaction):
    """Execution of one task per step from the list of the Barabási system.

    The highest priority task is executed with probability `greed` and a random task otherwise. The executed task is
    replaced by a new task with a uniform random priority, and every other task ages by one step. The waiting time of
    the executed task is reported as `wait`, so the run records one waiting time per step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of tasks.

    Examples:
        >>> from skcomplex.systems.social.organizations import ExecutionInteraction
        >>> ExecutionInteraction('execution', target='tasks').target
        'tasks'
    """

    draws = True

    law_classes = (UpdateLaw,)

    reports: ClassVar[tuple[str, ...]] = ('wait',)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, s: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the execution rule."""
        super()._prepare(s)
        if isinstance(s, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(s, self.execute)]

    def execute(self, system: SystemView) -> None:
        """Execute one task and replace it with a new one.

        Args:
            system:
                The system.
        """
        tasks = cast(list[TaskView], system.get_agents(self.target))
        if not tasks:
            return
        priorities = [task.priority for task in tasks]
        greedy = system.rng.random() < tasks[0].greed
        chosen = int(np.argmax(priorities)) if greedy else int(system.rng.integers(len(tasks)))
        for index, task in enumerate(tasks):
            task.age += 1
            if index == chosen:
                task.wait, task.age, task.priority = task.age, 0.0, float(system.rng.random())
                system.report('wait', task.wait)


class PriorityQueuesSystem(BaseSystem):
    """The Barabási priority queue.

    Its waiting times follow a power law as the choice becomes greedy. The system has the group `tasks` and the
    execution `execution`, which reports the waiting time of the task it executes at every step.

    Args:
        n_tasks:
            The length of the list of tasks.

        greed:
            The probability that the highest priority task is executed rather than a random one.

        random_state:
            The seed or generator of the initial priorities.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.organizations import PriorityQueuesSystem
        >>> PriorityQueuesSystem(3, random_state=0).prepare().system_.elements[0].size
        3
    """

    field = 'social/organizations'
    topics = ('queues', 'agents', 'bursts', 'power law')
    references = ('Barabási 2005', 'Castellano, Fortunato, and Loreto 2009')
    until = 200

    def __init__(
        self,
        n_tasks: int = 2,
        *,
        greed: float = 0.99999,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_tasks = n_tasks
        self.greed = greed
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        priorities = np.random.default_rng(self.random_state).random(self.n_tasks)
        tasks = TaskElement('tasks', priority=priorities, greed=self.greed, size=self.n_tasks)
        self.system_ = System(
            self.label,
            elements=[tasks],
            interactions=[ExecutionInteraction('execution', target='tasks')],
        )
