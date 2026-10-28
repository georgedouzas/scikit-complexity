"""Insider threat models: the risk that employees sabotage their organization."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

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


class SupervisorElement(Element):
    """Supervisor of the insider sabotage model, who wanders between adjacent cells.

    Args:
        label:
            The label of the agent or group.

    Examples:
        >>> from skcomplex.models.social.organizations import SupervisorElement
        >>> group = GroupElement('supervisors', element=SupervisorElement, size=3)
        >>> group.label, group.element.__name__, group.size
        ('supervisors', 'SupervisorElement', 3)
    """

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the wandering rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(_wander)]


class EmployeeElement(Element):
    """Employee of the insider sabotage model, whose risk is the sum of five factors.

    The risk is `R = P + D + S + K + A`: the predisposition `P`, the disgruntlement `D` of predisposed employees
    whose daily feeling about the organization falls short of their expectation, the stress `S` raised by seeing a
    supervisor, the skill `K` that grows every day, and the fixed affect `A`. The expectation weighs the first, the
    mean, and the latest feeling. Every day an employee wanders to an adjacent cell and leaves with probability
    `turnover / 365`, replaced by a new hire.

    Args:
        label:
            The label of the agent or group.

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
        >>> from skcomplex.models.social.organizations import EmployeeElement
        >>> group = GroupElement('employees', element=EmployeeElement, size=10, values={'affect': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('employees', 'EmployeeElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'risk', 'color_range': (0.0, 5.0)}),
        *,
        predisposed: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        disgruntlement: Annotated[float, StateSpec()] = 0.0,
        stress: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        skill: Annotated[float, StateSpec()] = 0.0,
        affect: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        risk: Annotated[float, StateSpec()] = 0.5,
        days: Annotated[float, StateSpec(domain=NN)] = 0.0,
        first: Annotated[float, StateSpec()] = 0.0,
        mean_feeling: Annotated[float, StateSpec()] = 0.0,
        expectation: Annotated[float, StateSpec()] = 0.0,
        turnover: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        culture: Annotated[float, ParameterSpec()] = 0.8,
        change: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        primacy: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        consistency: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        recency: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        affect_weight: Annotated[float, ParameterSpec()] = 1.0,
        alpha_seen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        lambda_seen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha_unseen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        lambda_unseen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        skill_growth: Annotated[float, ParameterSpec()] = 0.001,
        skill_variance: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0005,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predisposed = predisposed
        self.disgruntlement = disgruntlement
        self.stress = stress
        self.skill = skill
        self.affect = affect
        self.risk = risk
        self.days = days
        self.first = first
        self.mean_feeling = mean_feeling
        self.expectation = expectation
        self.turnover = turnover
        self.culture = culture
        self.change = change
        self.primacy = primacy
        self.consistency = consistency
        self.recency = recency
        self.affect_weight = affect_weight
        self.alpha_seen = alpha_seen
        self.lambda_seen = lambda_seen
        self.alpha_unseen = alpha_unseen
        self.lambda_unseen = lambda_unseen
        self.skill_growth = skill_growth
        self.skill_variance = skill_variance

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the daily rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.work)]

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
    employees = GroupElement(
        'employees',
        element=EmployeeElement,
        size=n_employees,
        values={'predisposed': predisposed, 'affect': affect, 'risk': predisposed + affect, 'turnover': turnover},
    )
    supervisors = GroupElement('supervisors', element=SupervisorElement, size=n_supervisors)
    return System(
        elements=[employees, supervisors],
        space=LatticeSpace((size, size), boundary='periodic', neighborhood='von_neumann'),
    )
