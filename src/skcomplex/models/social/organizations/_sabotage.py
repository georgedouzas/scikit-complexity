"""Insider threat models: the risk that employees sabotage their organization."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import AgentType, GridSpace, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

DAYS_PER_YEAR = 365
PREDISPOSED = 0.1
AFFECTS = (0.0, 0.5, 1.0)
AFFECT_SHARES = (0.1, 0.8, 0.1)


def _wander(agent: AgentView, model: ModelView) -> None:
    """Move an agent to a random adjacent cell."""
    cells = agent.neighbor_cells()
    agent.move_to(cells[int(model.rng.integers(len(cells)))])


def _feel(agent: AgentView, model: ModelView) -> None:
    """Draw how a predisposed employee feels about the organization today and update its disgruntlement."""
    feeling = model.rng.normal(agent.culture, np.sqrt(agent.change))
    if agent.days == 0:
        agent.first = feeling
    else:
        agent.disgruntlement = (agent.expectation - feeling) * agent.affect_weight
    agent.days += 1
    agent.mean_feeling += (feeling - agent.mean_feeling) / agent.days
    weights = agent.primacy + agent.consistency + agent.recency
    agent.expectation = (
        agent.primacy * agent.first + agent.consistency * agent.mean_feeling + agent.recency * feeling
    ) / weights


def _stress(agent: AgentView) -> None:
    """Raise the stress of an employee who sees an agent of another group, a supervisor, and let it decay otherwise."""
    supervised = any(other.group != agent.group for other in [*agent.neighbors(), *agent.agents_here()])
    if supervised:
        agent.stress += agent.alpha_seen * (agent.lambda_seen - agent.stress)
    else:
        agent.stress += agent.alpha_unseen * (agent.lambda_unseen - agent.stress)


def _rehire(agent: AgentView, model: ModelView) -> None:
    """Replace a leaving employee by a new hire with the same affect."""
    agent.predisposed = float(model.rng.random() < PREDISPOSED)
    agent.disgruntlement = agent.stress = agent.skill = 0.0
    agent.days = agent.mean_feeling = agent.first = agent.expectation = 0.0


class Supervisor(AgentType):
    """Supervisor of the insider sabotage model, who wanders between adjacent cells.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

    Examples:
        >>> from skcomplex.models.social.organizations import Supervisor
        >>> Supervisor.group('supervisors', 3)
        Group('supervisors', element=Supervisor, size=3)
    """

    def laws(self, s: Symbols) -> list[Law]:
        """Return the wandering rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(_wander)]


class Employee(AgentType):
    """Employee of the insider sabotage model, whose risk is the sum of five factors.

    The risk is `R = P + D + S + K + A`: the predisposition `P`, the disgruntlement `D` of predisposed employees
    whose daily feeling about the organization falls short of their expectation, the stress `S` raised by seeing a
    supervisor, the skill `K` that grows every day, and the fixed affect `A`. The expectation weighs the first, the
    mean, and the latest feeling. Every day an employee wanders to an adjacent cell and leaves with probability
    `turnover / 365`, replaced by a new hire.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        predisposed:
            Whether the employee is predisposed to sabotage, 1, or not, 0.

        disgruntlement:
            The initial disgruntlement `D`.

        stress:
            The initial stress `S`.

        skill:
            The initial skill `K`.

        affect:
            The affect `A`.

        risk:
            The initial risk `R`.

        days:
            The number of days the employee has felt about the organization.

        first:
            The first feeling.

        mean_feeling:
            The mean feeling.

        expectation:
            The expected feeling.

        turnover:
            The yearly share of employees who leave.

        culture:
            The mean daily feeling about the organization.

        change:
            The variance of the daily feeling.

        primacy:
            The weight of the first feeling in the expectation.

        consistency:
            The weight of the mean feeling in the expectation.

        recency:
            The weight of the latest feeling in the expectation.

        affect_weight:
            The factor of the shortfall of a feeling in the disgruntlement.

        alpha_seen:
            The learning rate of stress when a supervisor is in sight.

        lambda_seen:
            The level stress approaches when a supervisor is in sight.

        alpha_unseen:
            The learning rate of stress otherwise.

        lambda_unseen:
            The level stress approaches otherwise.

        skill_growth:
            The mean daily growth of skill.

        skill_variance:
            The variance of the daily growth of skill.

    Examples:
        >>> from skcomplex.models.social.organizations import Employee
        >>> Employee.group('employees', 10, affect=0.5)
        Group('employees', element=Employee, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'risk', 'color_range': (0.0, 5.0)})
    predisposed: float = state(default=0.0, domain=Domain.BOOLEAN)
    disgruntlement: float = state(default=0.0)
    stress: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    skill: float = state(default=0.0)
    affect: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    risk: float = state(default=0.5)
    days: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    first: float = state(default=0.0)
    mean_feeling: float = state(default=0.0)
    expectation: float = state(default=0.0)
    turnover: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    culture: float = parameter(default=0.8)
    change: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    primacy: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    consistency: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    recency: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    affect_weight: float = parameter(default=1.0)
    alpha_seen: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    lambda_seen: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    alpha_unseen: float = parameter(default=0.25, domain=Domain.NONNEGATIVE)
    lambda_unseen: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    skill_growth: float = parameter(default=0.001)
    skill_variance: float = parameter(default=0.0005, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the daily rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.work)]

    def work(self, agent: AgentView, model: ModelView) -> None:
        """Wander, feel, get stressed, learn, maybe leave, and update the risk.

        Args:
            agent:
                The employee.

            model:
                The model.
        """
        _wander(agent, model)
        if agent.predisposed:
            _feel(agent, model)
        _stress(agent)
        agent.skill += model.rng.normal(agent.skill_growth, np.sqrt(agent.skill_variance))
        if model.rng.random() < agent.turnover / DAYS_PER_YEAR:
            _rehire(agent, model)
        agent.risk = agent.predisposed + agent.disgruntlement + agent.stress + agent.skill + agent.affect


@register_model(
    field='social/organizations',
    tags=('organizations', 'agents', 'grid', 'insider threat'),
    references=('Ardiles Cruz et al. 2019', 'Carmichael et al. 2019'),
    defaults={'until': 365},
)
def insider_sabotage(
    size: int = 20,
    *,
    n_employees: int = 130,
    n_supervisors: int = 3,
    turnover: float = 0.15,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the insider sabotage model of employees whose risk of sabotage builds up over the days of a year.

    Args:
        size:
            The number of rows and columns of the torus.

        n_employees:
            The number of employees.

        n_supervisors:
            The number of supervisors.

        turnover:
            The yearly share of employees who leave.

        random_state:
            The seed or generator of the predispositions and affects.

    Returns:
        system:
            The model, with the groups `employees` and `supervisors`.

    Examples:
        >>> from skcomplex.models.social.organizations import insider_sabotage
        >>> [group.size for group in insider_sabotage(n_employees=20, n_supervisors=2).elements]
        [20, 2]
    """
    rng = np.random.default_rng(random_state)
    predisposed = (rng.random(n_employees) < PREDISPOSED).astype(float)
    affect = rng.choice(AFFECTS, n_employees, p=AFFECT_SHARES)
    employees = Employee.group(
        'employees',
        n_employees,
        predisposed=predisposed,
        affect=affect,
        risk=predisposed + affect,
        turnover=turnover,
    )
    supervisors = Supervisor.group('supervisors', n_supervisors)
    return System(
        components=[employees, supervisors],
        space=GridSpace(size, size, neighborhood='von_neumann', torus=True),
    )
