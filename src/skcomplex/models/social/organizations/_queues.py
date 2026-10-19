"""Queue models: tasks executed by priority, whose waiting times are heavy tailed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import ModelView


class Task(AgentType):
    """Task of the Barabási priority queue, with a priority, an age, and the waiting time when it was last executed.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        priority:
            The priority, uniform on 0 to 1 for a new task.

        age:
            The number of steps the task has waited.

        wait:
            The waiting time of the last task executed from this slot of the list.

        greed:
            The probability that the highest priority task is executed rather than a random one.

    Examples:
        >>> from skcomplex.models.social.organizations import Task
        >>> Task.group('tasks', 2, priority=[0.2, 0.7])
        Group('tasks', element=Task, size=2)
    """

    priority: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    age: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    wait: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    greed: float = parameter(default=0.99999, domain=Domain.NONNEGATIVE)


class Execution(Interaction):
    """Execution of one task per step from the list of the Barabási model.

    The highest priority task is executed with probability `greed` and a random task otherwise. The executed task is
    replaced by a new task with a uniform random priority, and every other task ages by one step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of tasks.

    Examples:
        >>> from skcomplex.models.social.organizations import Execution
        >>> Execution('execution', target='tasks').target
        'tasks'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the execution rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.execute)]

    def execute(self, model: ModelView) -> None:
        """Execute one task and replace it with a new one.

        Args:
            model:
                The model.
        """
        tasks = model.agents(self.target)
        if not tasks:
            return
        priorities = [task.priority for task in tasks]
        greedy = model.rng.random() < tasks[0].greed
        chosen = int(np.argmax(priorities)) if greedy else int(model.rng.integers(len(tasks)))
        for index, task in enumerate(tasks):
            task.age += 1
            if index == chosen:
                task.wait, task.age, task.priority = task.age, 0.0, float(model.rng.random())


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
    random_state: int | np.random.Generator | None = None,
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
            The model, with the group `tasks`.

    Examples:
        >>> from skcomplex.models.social.organizations import priority_queues
        >>> priority_queues(3, random_state=0).elements[0].size
        3
    """
    priorities = np.random.default_rng(random_state).random(n_tasks)
    tasks = Task.group('tasks', n_tasks, priority=priorities, greed=greed)
    return System(elements=[tasks], interactions=[Execution('execution', target='tasks')])
