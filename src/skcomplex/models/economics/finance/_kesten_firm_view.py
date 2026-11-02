"""Define the view of a firm of Kesten's process, as its growth rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class KestenFirmView(AgentView):
    """Firm of Kesten's process as its rule sees it: its size `x` and the bounds of its random factor and term."""

    x: float
    lambda_low: float
    lambda_high: float
    b_low: float
    b_high: float
