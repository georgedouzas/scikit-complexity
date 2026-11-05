"""Define the view of the pool of predictors of the El Farol bar, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class PredictorPoolView(AgentView):
    """Pool of predictors as the rules of the bar see it, with the `forecasts` and the `errors` of its predictors."""

    forecasts: Array
    errors: Array
