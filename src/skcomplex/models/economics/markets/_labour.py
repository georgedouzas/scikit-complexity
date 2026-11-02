"""Markets models: a labour market of firms posting wage offers and unemployed workers accepting the best one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MapInteraction, MatchingInteraction, TransitionInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
    ObservableSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StableMatchingSpec,
)
from ..._catalog import register_model
from ._employer import EmployerElement
from ._employer_view import EmployerView
from ._jobseeker import JobseekerElement
from ._jobseeker_view import JobseekerView


def hire(trade: MatchSpec, model: ModelView) -> None:
    """Record the employer, the wage, and the steps of the hire and of its offer on the hired worker.

    The hired worker keeps its search flag `supply` until the next step, which the market lowered when it cleared, so
    the observable `searching` counts every worker who searched in the step.

    Args:
        trade:
            The match of a firm, the requester of labour, with a worker, its provider.

        model:
            The model view of the market.
    """
    worker, firm = trade.provider, trade.requester
    if not isinstance(worker, JobseekerView) or not isinstance(firm, EmployerView):
        return
    worker.employer, worker.wage, worker.duration = float(firm.id), trade.value, 0.0
    worker.supply += trade.quantity
    worker.hired, worker.offer = model.time, firm.posted


def _unemployment(employers: object) -> float:
    """Return the share of unemployed workers, whose employer is negative."""
    return float(np.mean(np.asarray(employers, dtype=float) < 0))


@register_model(
    field='economics/markets',
    tags=('labour', 'market', 'matching', 'agents'),
    references=('Hamill and Gilbert 2016 ch. 7',),
    defaults={'until': 200},
    observables=lambda system: [
        ObservableSpec('unemployment', group='workers', variable='employer', reduce=_unemployment),
        ObservableSpec('searching', group='workers', variable='supply', reduce='sum'),
    ],
)
def labour_market(
    n_workers: int = 200,
    n_firms: int = 40,
    *,
    separation: float = 0.05,
    vacancy_rate: float = 0.3,
    reservation: float = 1.0,
    decline: float = 0.1,
    low_wage: float = 0.5,
    high_wage: float = 1.0,
    unemployment: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Hamill and Gilbert's labour market of firms posting wage offers and unemployed workers searching.

    In the phase of offers, jobs are destroyed at the separation rate `s`, every firm posts a vacancy with a uniform
    wage offer at the vacancy rate, and unemployed workers set reservation wages that fall with the duration of their
    search. In the phase of matching, a deferred-acceptance market pairs vacancies with workers whose reservation wage
    they cover, every worker getting the best offer of a stable matching. The unemployment rate settles at the
    flow-balance value `s / (s + f)`, with `f` the share of searching workers hired in a step.

    Args:
        n_workers:
            The number of workers.

        n_firms:
            The number of firms, each with at most one vacancy per step.

        separation:
            The probability `s` that a job is destroyed in a step.

        vacancy_rate:
            The probability that a firm posts a vacancy in a step.

        reservation:
            The reservation wage at the start of a search.

        decline:
            The relative fall of the reservation wage per step of search.

        low_wage:
            The lowest wage offer.

        high_wage:
            The highest wage offer.

        unemployment:
            The initial share of unemployed workers.

        random_state:
            The seed or generator of the initial employment.

    Returns:
        system:
            The model, with the groups `firms` and `workers`, the market `labour` with the reporters `labour.value`
            (the mean wage of the hires) and `labour.volume` (the hires), and the catalog observables of the
            unemployment rate `unemployment` and of the searching workers `searching`.

    Examples:
        >>> from skcomplex.models.economics.markets import labour_market
        >>> [group.size for group in labour_market(20, 5).elements]
        [5, 20]
    """
    rng = np.random.default_rng(random_state)
    employed = rng.random(n_workers) >= unemployment
    employers = np.where(employed, rng.integers(n_firms, size=n_workers), -1).astype(float)
    firms = GroupElement(
        'firms',
        element=EmployerElement,
        size=n_firms,
        values={'vacancy_rate': vacancy_rate, 'low_wage': low_wage, 'high_wage': high_wage, 'bid': low_wage},
    )
    workers = GroupElement(
        'workers',
        element=JobseekerElement,
        size=n_workers,
        values={
            'employer': employers,
            'wage': np.where(employed, (low_wage + high_wage) / 2, 0.0),
            'supply': 0.0,
            'ask': reservation,
            'separation': separation,
            'reservation': reservation,
            'decline': decline,
        },
    )
    offers = [
        MapInteraction(f'{group}_{name}', target=group, state=name, function=function, form='agent', phase='offers')
        for group, steps in (('firms', EmployerElement.get_posting()), ('workers', JobseekerElement.get_search()))
        for name, function in steps
    ]
    losses = TransitionInteraction(
        'workers_separation',
        target='workers',
        state='employer',
        source=lambda value: value >= 0,
        destination=-1.0,
        probability=lambda agent, model: agent.separation,
        form='agent',
        phase='offers',
    )
    spells = [
        MapInteraction(f'workers_{name}', target='workers', state=name, function=function, form='agent', phase='offers')
        for name, function in JobseekerElement.get_spell()
    ]
    market = MatchingInteraction(
        'labour',
        requesters='firms',
        providers='workers',
        rule=StableMatchingSpec(clearing='requester'),
        limit=('bid', 'ask'),
        quantity=('demand', 'supply'),
        phase='matching',
        on_match=hire,
    )
    return System(
        elements=[firms, workers],
        interactions=[
            *offers,
            losses,
            *spells,
            market,
        ],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('offers'), PhaseSpec('matching')])),
    )
