"""Insider threat systems: the risk that employees sabotage their organization."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import LifeCycleInteraction, MapInteraction, MovementInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.components.types import SystemViewProtocol
from ..._catalog import register_system
from ._employee import EmployeeElement
from ._employee_view import EmployeeView
from ._supervisor import SupervisorElement

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
