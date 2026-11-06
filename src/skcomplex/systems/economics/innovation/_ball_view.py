"""Define the view of the balls of one colour in an urn, as the urn sees them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class BallView(AgentView):
    """Balls of one colour as the urn sees them: their `color`, their `count`, and whether the node `drawn` it."""

    color: float
    count: float
    drawn: float
