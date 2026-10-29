"""Define the view of a consumer of the fair food market, as its shopping rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ConsumerView(AgentView):
    """Consumer as the shopping rule sees it: its choice `fair`, `ethics`, and `budget`, and its parameters.

    The parameters are the `premium` of fair food, the weights `ethics_weight`, `price_weight`, and `social_weight`
    of the utility, and the `intensity` of the logit choice.
    """

    fair: float
    ethics: float
    budget: float
    premium: float
    ethics_weight: float
    price_weight: float
    social_weight: float
    intensity: float
