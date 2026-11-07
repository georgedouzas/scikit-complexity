"""Insider threat systems: the risk that employees sabotage their organization."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction, MapInteraction, MovementInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class SupervisorElement(Element):
    """Supervisor of the insider sabotage system, who wanders between adjacent cells.

    The wandering is a `MovementInteraction` to a random adjacent cell, which `insider_sabotage` adds.

    Args:
        label:
            The label of the agent or group.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import SupervisorElement
        >>> group = GroupElement('supervisors', element=SupervisorElement, size=3)
        >>> group.label, group.element.__name__, group.size
        ('supervisors', 'SupervisorElement', 3)
    """


class EmployeeView(AgentView):
    """View of an employee of the insider sabotage system, with its states and parameters as attributes."""

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


class EmployeeElement(Element):
    """Employee of the insider sabotage system, whose risk is the sum of five factors.

    The risk is `R = P + D + S + K + A`: the predisposition `P`, the disgruntlement `D` of predisposed employees
    whose daily feeling about the organization falls short of their expectation, the stress `S` raised by seeing a
    supervisor, the skill `K` that grows every day, and the fixed affect `A`. The expectation weighs the first, the
    mean, and the latest feeling. Every day an employee wanders to an adjacent cell and leaves with probability
    `turnover / 365`, replaced by a new hire. In `insider_sabotage`, a movement, a map of every state, and a life cycle
    that renews the employees who leave run these steps in turn in the activation of every employee.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

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

        feeling:
            The latest feeling.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import EmployeeElement
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
        feeling: Annotated[float, StateSpec()] = 0.0,
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
        self.feeling = feeling
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

    @classmethod
    def _get_view(cls) -> type[EmployeeView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return EmployeeView


DAYS_PER_YEAR = 365


PREDISPOSED = 0.1


AFFECTS = (0.0, 0.5, 1.0)


AFFECT_SHARES = (0.1, 0.8, 0.1)


HIRE_RESETS = ('disgruntlement', 'stress', 'skill', 'days', 'mean_feeling', 'first', 'expectation')


def _feel(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the feeling of a predisposed employee about the organization today, drawn, or the last one otherwise."""
    if not agent.predisposed:
        return float(agent.feeling)
    return float(system.rng.normal(agent.culture, np.sqrt(agent.change)))


def _remember_first(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the first feeling of an employee, today's feeling on the first day of a predisposed employee."""
    return float(agent.feeling if agent.predisposed and agent.days == 0 else agent.first)


def _disgruntle(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the disgruntlement, the shortfall of today's feeling from the expectation of a predisposed employee."""
    if not agent.predisposed or agent.days == 0:
        return float(agent.disgruntlement)
    return float((agent.expectation - agent.feeling) * agent.affect_weight)


def _count_day(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return one day felt for a predisposed employee, and none otherwise."""
    return float(agent.predisposed)


def _average(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the change of the mean feeling of a predisposed employee by today's feeling."""
    return float((agent.feeling - agent.mean_feeling) / agent.days) if agent.predisposed else 0.0


def _expect(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the expectation of a predisposed employee, which weighs the first, the mean, and today's feeling."""
    if not agent.predisposed:
        return float(agent.expectation)
    weights = agent.primacy + agent.consistency + agent.recency
    weighted = agent.primacy * agent.first + agent.consistency * agent.mean_feeling + agent.recency * agent.feeling
    return float(weighted / weights)


def _stress(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the change of stress towards its level in sight of a supervisor, an agent of another group, or out."""
    supervised = any(other.group != agent.group for other in [*agent.neighbors(), *agent.agents_here()])
    if supervised:
        return float(agent.alpha_seen * (agent.lambda_seen - agent.stress))
    return float(agent.alpha_unseen * (agent.lambda_unseen - agent.stress))


def _learn(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the growth of the skill of an employee in a day."""
    return float(system.rng.normal(agent.skill_growth, np.sqrt(agent.skill_variance)))


def _leave(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the daily probability that an employee leaves."""
    return float(agent.turnover / DAYS_PER_YEAR)


def _hire(agent: EmployeeView | None, system: SystemViewProtocol) -> dict[str, object]:
    """Return the values of a new hire who replaces a leaving employee and keeps its affect."""
    return {'predisposed': float(system.rng.random() < PREDISPOSED), **dict.fromkeys(HIRE_RESETS, 0.0)}


def _sum_risk(agent: EmployeeView, system: SystemViewProtocol) -> float:
    """Return the risk, the sum of the predisposition, the disgruntlement, the stress, the skill, and the affect."""
    return float(agent.predisposed + agent.disgruntlement + agent.stress + agent.skill + agent.affect)


@register_system(
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
    """Return the insider sabotage system of employees whose risk of sabotage builds up over the days of a year.

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
            The system, with the groups `employees` and `supervisors`.

    Examples:
        >>> from skcomplex.systems.social.organizations import insider_sabotage
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
    wandering = MovementInteraction('wandering', target='supervisors', schedule='member')
    maps = [
        ('feeling', _feel, 'replace'),
        ('first', _remember_first, 'replace'),
        ('disgruntlement', _disgruntle, 'replace'),
        ('days', _count_day, 'increment'),
        ('mean_feeling', _average, 'increment'),
        ('expectation', _expect, 'replace'),
        ('stress', _stress, 'increment'),
        ('skill', _learn, 'increment'),
    ]
    work = [
        MovementInteraction('working', target='employees', schedule='member'),
        *(
            MapInteraction(
                state,
                target='employees',
                state=state,
                function=function,
                mode=mode,
                schedule='member',
                form='agent',
            )
            for state, function, mode in maps
        ),
        LifeCycleInteraction('turnover', target='employees', death=_leave, renew=True, values=_hire, schedule='member'),
        MapInteraction('risk', target='employees', state='risk', function=_sum_risk, schedule='member', form='agent'),
    ]
    return System(
        elements=[employees, supervisors],
        interactions=[wandering, *work],
        space=LatticeSpace((size, size), boundary='periodic', neighborhood='von_neumann'),
    )
