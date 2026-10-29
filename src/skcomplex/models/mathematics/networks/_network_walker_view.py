"""View of a walker on a network, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class NetworkWalkerView(AgentView):
    """View of a walker on a network, with its states as attributes."""

    steps: float
