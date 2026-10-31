"""Markets models: a labour market of firms posting wage offers and unemployed workers accepting the best one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import FeedbackInteraction, MatchingInteraction, StableMatchingProtocol
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from ..._catalog import register_model
from ._employer_view import EmployerView
from ._jobseeker_view import JobseekerView


class EmployerElement(Element):
    """Firm that posts a vacancy with a random wage offer in some steps, filled by at most one worker.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.economics.markets import EmployerElement
        >>> GroupElement('firms', element=EmployerElement, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        demand: Annotated[float, StateSpec(domain=NN)] = 0.0,
        posted: Annotated[float, StateSpec()] = -1.0,
        vacancy_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        low_wage: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        high_wage: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bid = bid
        self.demand = demand
        self.posted = posted
        self.vacancy_rate = vacancy_rate
        self.low_wage = low_wage
        self.high_wage = high_wage

    @classmethod
    def _get_view(cls) -> type[EmployerView]:
        """Return the view of an employer, with its vacancy and the parameters of its offers."""
        return EmployerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the posting rule in the phase of offers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.post, phase='offers')]

    def post(self, agent: EmployerView, model: ModelView) -> None:
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
            agent.bid = float(model.rng.uniform(agent.low_wage, agent.high_wage))
            agent.posted = model.time


class JobseekerElement(Element):
    """Worker who loses its job at the separation rate and, while unemployed, lowers its reservation wage.

    A worker who loses its job searches from the next step. An unemployed worker asks the reservation wage
    `w_0 (1 - decline)^d`, where `d` is the number of steps of search, and is hired by the matching market.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.economics.markets import JobseekerElement
        >>> GroupElement('workers', element=JobseekerElement, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        supply: Annotated[float, StateSpec(domain=NN)] = 1.0,
        employer: Annotated[float, StateSpec()] = -1.0,
        wage: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        duration: Annotated[float, StateSpec(domain=NN)] = 0.0,
        hired: Annotated[float, StateSpec()] = -1.0,
        offer: Annotated[float, StateSpec()] = -1.0,
        separation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        reservation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        decline: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.ask = ask
        self.supply = supply
        self.employer = employer
        self.wage = wage
        self.duration = duration
        self.hired = hired
        self.offer = offer
        self.separation = separation
        self.reservation = reservation
        self.decline = decline

    @classmethod
    def _get_view(cls) -> type[JobseekerView]:
        """Return the view of a worker, with its search and job states and their parameters."""
        return JobseekerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of separations and search in the phase of offers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.search, phase='offers')]

    def search(self, agent: JobseekerView, model: ModelView) -> None:
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


def hire(trade: MatchSpec, model: ModelView) -> None:
    """Record the employer, the wage, and the steps of the hire and of its offer on the hired worker.

    Args:
        trade:
            The match of a firm, the requester of labour, with a worker, its provider.

        model:
            The model view of the market.
    """
    worker, firm = trade.provider, trade.requester
    if not isinstance(worker, JobseekerView) or not isinstance(firm, EmployerView):
        return
    worker.employer, worker.wage, worker.duration = float(firm.id), trade.price, 0.0
    worker.hired, worker.offer = model.time, firm.posted


class SearchCountInteraction(FeedbackInteraction[ModelView]):
    """Statistics office that counts the workers searching in a step before the labour market clears.

    Args:
        label:
            The label of the institution.

        rule:
            Unused, since the office overrides `act`.

        phase:
            The phase the office counts in, which must precede the clearing of the market.

        workers:
            The path of the group of workers.

    Examples:
        >>> from skcomplex.models.economics.markets import SearchCountInteraction
        >>> SearchCountInteraction('office', workers='workers', phase='matching').reports
        ('searching',)
    """

    reports: ClassVar[tuple[str, ...]] = ('searching',)

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[ModelView], None] | None = None,
        phase: str | None = None,
        workers: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.workers = workers

    def act(self, model: ModelView) -> None:
        """Report the number of searching workers.

        Args:
            model:
                The model view of the office.
        """
        model.report('searching', sum(worker.supply > 0 for worker in model.agents('workers', JobseekerView)))


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
    market = MatchingInteraction(
        'labour',
        requesters='firms',
        providers='workers',
        protocol=StableMatchingProtocol(pricing='requester'),
        limit=('bid', 'ask'),
        quantity=('demand', 'supply'),
        phase='matching',
        on_match=hire,
    )
    return System(
        elements=[firms, workers],
        interactions=[SearchCountInteraction('office', workers='workers', phase='matching'), market],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('offers'), PhaseSpec('matching')])),
    )
