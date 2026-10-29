"""Queue models: tasks executed by priority, whose waiting times are heavy tailed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._task_view import TaskView


class TaskElement(Element):
    """Task of the Barabási priority queue, with a priority, an age, and the waiting time when it was last executed.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        priority:
            The priority, uniform on 0 to 1 for a new task.

        age:
            The number of steps the task has waited.

        wait:
            The waiting time of the last task executed from this slot of the list.

        greed:
            The probability that the highest priority task is executed rather than a random one.

    Examples:
        >>> from skcomplex.models.social.organizations import TaskElement
        >>> group = GroupElement('tasks', element=TaskElement, size=2, values={'priority': [0.2, 0.7]})
        >>> group.label, group.element.__name__, group.size
        ('tasks', 'TaskElement', 2)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('histograms', {'color_by': 'age', 'shown': ('agents',)}),
        *,
        priority: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        age: Annotated[float, StateSpec(domain=NN)] = 0.0,
        wait: Annotated[float, StateSpec(domain=NN)] = 0.0,
        greed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.99999,
    ) -> None:
        super().__init__(label, visual=visual)
        self.priority = priority
        self.age = age
        self.wait = wait
        self.greed = greed

    @classmethod
    def _get_view(cls) -> type[TaskView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return TaskView


class ExecutionInteraction(Interaction):
    """Execution of one task per step from the list of the Barabási model.

    The highest priority task is executed with probability `greed` and a random task otherwise. The executed task is
    replaced by a new task with a uniform random priority, and every other task ages by one step. The waiting time of
    the executed task is reported as `wait`, so the run records one waiting time per step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of tasks.

    Examples:
        >>> from skcomplex.models.social.organizations import ExecutionInteraction
        >>> ExecutionInteraction('execution', target='tasks').target
        'tasks'
    """

    reports: ClassVar[tuple[str, ...]] = ('wait',)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the execution rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.execute)]

    def execute(self, model: ModelView) -> None:
        """Execute one task and replace it with a new one.

        Args:
            model:
                The model.
        """
        tasks = model.agents(self.target, TaskView)
        if not tasks:
            return
        priorities = [task.priority for task in tasks]
        greedy = model.rng.random() < tasks[0].greed
        chosen = int(np.argmax(priorities)) if greedy else int(model.rng.integers(len(tasks)))
        for index, task in enumerate(tasks):
            task.age += 1
            if index == chosen:
                task.wait, task.age, task.priority = task.age, 0.0, float(model.rng.random())
                model.report('wait', task.wait)


@register_model(
    field='social/organizations',
    tags=('queues', 'agents', 'bursts', 'power law'),
    references=('Barabási 2005', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 200},
)
def priority_queues(
    n_tasks: int = 2,
    *,
    greed: float = 0.99999,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Barabási priority queue, whose waiting times follow a power law as the choice becomes greedy.

    Args:
        n_tasks:
            The length of the list of tasks.

        greed:
            The probability that the highest priority task is executed rather than a random one.

        random_state:
            The seed or generator of the initial priorities.

    Returns:
        system:
            The model, with the group `tasks` and the execution `execution`, which reports the waiting time of the
            task it executes at every step.

    Examples:
        >>> from skcomplex.models.social.organizations import priority_queues
        >>> priority_queues(3, random_state=0).elements[0].check().prepare().size_
        3
    """
    priorities = np.random.default_rng(random_state).random(n_tasks)
    tasks = GroupElement('tasks', element=TaskElement, size=n_tasks, values={'priority': priorities, 'greed': greed})
    return System(elements=[tasks], interactions=[ExecutionInteraction('execution', target='tasks')])
