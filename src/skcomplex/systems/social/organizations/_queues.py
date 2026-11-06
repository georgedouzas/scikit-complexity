"""Queue systems: tasks executed by priority, whose waiting times are heavy tailed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ._execution import ExecutionInteraction
from ._task import TaskElement


@register_system(
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
            The system, with the group `tasks` and the execution `execution`, which reports the waiting time of the
            task it executes at every step.

    Examples:
        >>> from skcomplex.systems.social.organizations import priority_queues
        >>> priority_queues(3, random_state=0).elements[0].prepare().size_
        3
    """
    priorities = np.random.default_rng(random_state).random(n_tasks)
    tasks = GroupElement('tasks', element=TaskElement, size=n_tasks, values={'priority': priorities, 'greed': greed})
    return System(elements=[tasks], interactions=[ExecutionInteraction('execution', target='tasks')])
