"""Execution of one task per step from the list of the Barabási system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ._task_view import TaskView


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

    def execute(self, system: SystemView) -> None:
        """Execute one task and replace it with a new one.

        Args:
            system:
                The system.
        """
        tasks = system.agents(self.target, TaskView)
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
