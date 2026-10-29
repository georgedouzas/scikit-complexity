"""Define the view of a car at the car wash, as the workers and the operation see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class CarView(AgentView):
    """Car as the rules see it: its `stage`, the `work` left on it, and the size of its `crew`."""

    stage: float
    work: float
    crew: float
