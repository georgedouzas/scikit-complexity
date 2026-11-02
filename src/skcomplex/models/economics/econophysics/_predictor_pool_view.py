"""Define the view of the pool of predictors of the El Farol bar, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import Array


class PredictorPoolView(AgentView):
    """Pool of predictors as the rules of the bar see it, with the `forecasts` and the `errors` of its predictors."""

    forecasts: Array
    errors: Array
