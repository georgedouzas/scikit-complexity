"""View of an employee of the insider sabotage model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class EmployeeView(AgentView):
    """View of an employee of the insider sabotage model, with its states and parameters as attributes."""

    predisposed: float
    disgruntlement: float
    stress: float
    skill: float
    affect: float
    risk: float
    days: float
    first: float
    mean_feeling: float
    expectation: float
    feeling: float
    turnover: float
    culture: float
    change: float
    primacy: float
    consistency: float
    recency: float
    affect_weight: float
    alpha_seen: float
    lambda_seen: float
    alpha_unseen: float
    lambda_unseen: float
    skill_growth: float
    skill_variance: float
