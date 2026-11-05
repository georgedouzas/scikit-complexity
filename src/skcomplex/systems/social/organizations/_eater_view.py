"""View of a person of the obesity system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class EaterView(AgentView):
    """View of a person of the obesity system, with its states and parameters as attributes."""

    home: Array
    work: Array
    weight: float
    height: float
    age: float
    male: float
    habit: float
    calories: float
    bmi: float
    day: float
    reach: float
    activity: float
