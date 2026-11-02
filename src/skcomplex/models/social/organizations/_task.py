"""Task of the Barabási priority queue, with a priority, an age, and the waiting time when it was last executed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
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
        >>> from skcomplex.core.elements import GroupElement
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
