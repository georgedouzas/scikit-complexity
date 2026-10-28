"""A stepper with a fixed horizon is asked to advance beyond it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...errors import SolverError


class HorizonError(SolverError):
    """A stepper with a fixed horizon is asked to advance beyond it."""
