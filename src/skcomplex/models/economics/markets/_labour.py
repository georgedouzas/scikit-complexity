"""Markets models: a labour market of firms posting wage offers and unemployed workers accepting the best one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np

from ....core.base import Domain, parameter, role, state
from ....core.model import DiscreteTime, Element, Law, Observable, Phase, Phases, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ..realizations import DeferredAcceptance, Institution, Market, Trade


class Employer(Element):
    """Firm that posts a vacancy with a random wage offer in some steps, filled by at most one worker.

    Args:
        label:
            The label of the agent or group.

        bid:
            The wage offered for the vacancy.

        demand:
            The number of open vacancies, one or zero.

        posted:
            The step in which the vacancy was posted.

        vacancy_rate:
            The probability of posting a vacancy in a step.

        low_wage:
            The lowest wage offer.

        high_wage:
            The highest wage offer.

    Examples:
        >>> from skcomplex.models.economics.markets import Employer
        >>> Employer.group('firms', 5).size
        5
    """

    bid: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    demand: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    posted: float = state(default=-1.0)
    vacancy_rate: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    low_wage: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    high_wage: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the posting rule in the phase of offers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.post, phase='offers')]

    def post(self, agent: AgentView, model: ModelView) -> None:
        """Withdraw the last vacancy and post a new one with a uniform wage offer at the vacancy rate.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        agent.demand = 0.0
        if model.rng.random() < agent.vacancy_rate:
            agent.demand = 1.0
            agent.bid = model.rng.uniform(agent.low_wage, agent.high_wage)
            agent.posted = model.time


class Jobseeker(Element):
    """Worker who loses its job at the separation rate and, while unemployed, lowers its reservation wage.

    A worker who loses its job searches from the next step. An unemployed worker asks the reservation wage
    `w_0 (1 - decline)^d`, where `d` is the number of steps of search, and is hired by the matching market.

    Args:
        label:
            The label of the agent or group.

        ask:
            The reservation wage.

        supply:
            One while the worker searches in the current step, zero otherwise.

        employer:
            The id of the employer, or -1 for an unemployed worker.

        wage:
            The wage of the current job.

        duration:
            The number of steps of the current spell of search.

        hired:
            The step of the last hire.

        offer:
            The step in which the offer of the last hire was posted.

        separation:
            The probability that a job is destroyed in a step.

        reservation:
            The reservation wage at the start of a search.

        decline:
            The relative fall of the reservation wage per step of search.

    Examples:
        >>> from skcomplex.models.economics.markets import Jobseeker
        >>> Jobseeker.group('workers', 5).size
        5
    """

    ask: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    supply: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)
    employer: float = state(default=-1.0)
    wage: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    duration: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    hired: float = state(default=-1.0)
    offer: float = state(default=-1.0)
    separation: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    reservation: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    decline: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of separations and search in the phase of offers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.search, phase='offers')]

    def search(self, agent: AgentView, model: ModelView) -> None:
        """Lose the job at the separation rate, or search with a reservation wage that falls with the duration.

        Args:
            agent:
                The worker.

            model:
                The model.
        """
        if agent.employer >= 0:
            agent.supply = 0.0
            if model.rng.random() < agent.separation:
                agent.employer, agent.wage, agent.duration = -1.0, 0.0, 0.0
            return
        agent.supply = 1.0
        agent.ask = agent.reservation * (1 - agent.decline) ** agent.duration
        agent.duration += 1


def hire(trade: Trade, model: Any) -> None:  # noqa: ANN401
    """Record the employer, the wage, and the steps of the hire and of its offer on the hired worker.

    Args:
        trade:
            The match of a firm, the buyer of labour, with a worker, the seller.

        model:
            The model view of the market.
    """
    worker, firm = trade.seller, trade.buyer
    worker.employer, worker.wage, worker.duration = float(firm.id), trade.price, 0.0
    worker.hired, worker.offer = model.time, firm.posted


class SearchCount(Institution):
    """Statistics office that counts the workers searching in a step before the labour market clears.

    Args:
        label:
            The label of the institution.

        workers:
            The path of the group of workers.

        rule:
            Unused, since the office overrides `act`.

        phase:
            The phase the office counts in, which must precede the clearing of the market.

    Examples:
        >>> from skcomplex.models.economics.markets import SearchCount
        >>> SearchCount('office', workers='workers', phase='matching').reports
        ('searching',)
    """

    reports: ClassVar[tuple[str, ...]] = ('searching',)
    workers: str = role()

    def act(self, model: Any) -> None:  # noqa: ANN401
        """Report the number of searching workers.

        Args:
            model:
                The model view of the office.
        """
        model.report('searching', sum(worker.supply > 0 for worker in model.agents('workers')))


def _unemployed(agent: AgentView) -> float:
    """Return one for an unemployed worker and zero for an employed one."""
    return float(agent.employer < 0)


@register_model(
    field='economics/markets',
    tags=('labour', 'market', 'matching', 'agents'),
    references=('Hamill and Gilbert 2016 ch. 7',),
    defaults={'until': 200},
    observables=lambda system: [Observable('unemployment', _unemployed, group='workers')],
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
            The model, with the groups `firms` and `workers`, the market `labour` with the reporters `labour.price`
            (the mean wage of the hires) and `labour.volume` (the hires), the reporter `office.searching`, and the
            unemployment rate `unemployment`.

    Examples:
        >>> from skcomplex.models.economics.markets import labour_market
        >>> [group.size for group in labour_market(20, 5).elements]
        [5, 20]
    """
    rng = np.random.default_rng(random_state)
    employed = rng.random(n_workers) >= unemployment
    employers = np.where(employed, rng.integers(n_firms, size=n_workers), -1).astype(float)
    firms = Employer.group(
        'firms',
        n_firms,
        vacancy_rate=vacancy_rate,
        low_wage=low_wage,
        high_wage=high_wage,
        bid=low_wage,
    )
    workers = Jobseeker.group(
        'workers',
        n_workers,
        employer=employers,
        wage=np.where(employed, (low_wage + high_wage) / 2, 0.0),
        supply=0.0,
        ask=reservation,
        separation=separation,
        reservation=reservation,
        decline=decline,
    )
    market = Market(
        'labour',
        buyers='firms',
        sellers='workers',
        protocol=DeferredAcceptance(pricing='bid'),
        phase='matching',
        money=None,
        goods=None,
        on_trade=hire,
    )
    return System(
        components=[firms, workers],
        interactions=[SearchCount('office', workers='workers', phase='matching'), market],
        time=DiscreteTime(schedule=Phases([Phase('offers'), Phase('matching')])),
    )
