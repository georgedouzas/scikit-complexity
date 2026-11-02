"""Define the view of a firm of Gibrat's process, as its growth and the reflecting floor see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class GibratFirmView(AgentView):
    """Firm of Gibrat's process as the rules see it, with its size and growth factor.

    It reads its size `x`, its `previous` size, and the mean `mu` and the deviation `sigma` of its growth factor.
    """

    x: float
    previous: float
    mu: float
    sigma: float
