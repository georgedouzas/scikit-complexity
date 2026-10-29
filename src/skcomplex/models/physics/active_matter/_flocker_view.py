"""Define the view of the agents of a flocker group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class FlockerView(AgentView):
    """Flocker as rules see it, with its heading.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the model.
    """

    theta: float
