"""A stepper with a fixed horizon is asked to advance beyond it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...errors import SolverError


class HorizonError(SolverError):
    """An error raised when a solver with a fixed horizon is asked to go beyond it.

    A solver that draws its noise for the whole run in advance, such as fractional noise up to a horizon of 1, cannot
    move past that horizon, and raises it with the times. It guards the runs of rough prices, river flows, and network
    traffic.

    Read more in the user guide, under Troubleshooting.

    See Also:
        FractionalEulerMaruyamaSolver: Draws its noise up to a horizon.

    Examples:
        >>> from skcomplex.core.engine.solvers import HorizonError
        >>> from skcomplex.core.errors import SolverError
        >>> issubclass(HorizonError, SolverError)
        True
    """
