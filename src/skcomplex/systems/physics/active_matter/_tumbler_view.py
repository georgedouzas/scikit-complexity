"""Define the view of the agents of a run-and-tumble group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TumblerView(AgentView):
    """Run-and-tumble agent as rules see it, with its direction and the parameters of its group.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the system.
    """

    theta: float
    speed: float
    tumble_rate: float
    width: float
