"""Define the view of a firm of the toy business-cycle model, as its rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class InvestingFirmView(AgentView):
    """Firm as its rule sees it: its states `A`, `K`, `Z`, `B`, `Y`, `P` and parameters `gamma`, `phi`, `r`, `pbar`."""

    A: float
    K: float
    Z: float
    B: float
    Y: float
    P: float
    gamma: float
    phi: float
    r: float
    pbar: float
