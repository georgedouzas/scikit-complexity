"""View of an agent of the voter model variants, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class VariantVoterView(AgentView):
    """View of an agent of the voter model variants, with its states and parameters as attributes."""

    opinion: float
    contrarian: float
    threshold: float
