"""Define the view of an employer of the labour market, as its posting rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class EmployerView(AgentView):
    """Employer as the rules see it: its offer `bid`, vacancy `demand`, step `posted`, and its wage parameters.

    The parameters are the `vacancy_rate` and the range `low_wage` to `high_wage` of the wage offers.
    """

    bid: float
    demand: float
    posted: float
    vacancy_rate: float
    low_wage: float
    high_wage: float
