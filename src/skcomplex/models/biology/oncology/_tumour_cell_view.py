"""View of a tumour cell as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class TumourCellView(AgentView):
    """View of a tumour cell, with its states and parameters as attributes.

    `TumourCellElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    hypoxic: float
    consumption: float
    proliferation: float
    division: float
    hypoxia: float
    secretion: float
    necrosis: float
