"""Define the view of an urn on a node, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class UrnView(AgentView):
    """Urn as its rules see it: its `novelties`, the `color` of its last draw, and whether that draw was `novel`."""

    novelties: float
    color: float
    novel: float
