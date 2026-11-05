"""View of a fragment, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class FragmentView(AgentView):
    """View of a fragment, with its states and parameters as attributes."""

    length: float
    rate: float
