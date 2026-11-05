"""Define the view of a patron of the El Farol bar, as the rules of the bar see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PatronView(AgentView):
    """Patron as the rules of the bar see it, with its predictors, forecast, and attendance.

    It reads the bit mask of its `predictors`, its `forecast`, whether it `attends`, and the `capacity` of the bar.
    """

    predictors: float
    forecast: float
    attends: float
    capacity: float
