"""Define the view of a household of the phone adoption model, as its rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class HouseholdView(AgentView):
    """Household as its rule sees it, with its income, phone, and class.

    It reads its `income`, its `phone`, its `social_class`, the `growth` of its income, and the `threshold` of
    affording a phone.
    """

    income: float
    phone: float
    social_class: float
    growth: float
    threshold: float
