"""Define the view of the agents of a diffusion-limited aggregation walker group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class WalkerView(AgentView):
    """Walker of diffusion-limited aggregation as rules see it, with whether it belongs to the aggregate.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the model.
    """

    stuck: float
