"""Markets systems: a labour market of firms posting wage offers and unemployed workers accepting the best one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction, MatchingInteraction, TransitionInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import (
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StableMatchingSpec,
    StateSpec,
    VisualSpec,
)


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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import EmployerElement
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

    @staticmethod
    def get_posting() -> tuple[tuple[str, Callable[[EmployerView, SystemView], float]], ...]:
        """Return the states a firm sets in the phase of offers, in their order.

        The firm withdraws its last vacancy and posts a new one at the vacancy rate, with a uniform wage offer and the
        step of posting.

        Returns:
            posting:
                The name of every state and the function of its value, of the firm and the system view.
        """
        return (
            ('demand', lambda agent, system: float(system.rng.random() < agent.vacancy_rate)),
            (
                'bid',
                lambda agent, system: (
                    float(system.rng.uniform(agent.low_wage, agent.high_wage)) if agent.demand else agent.bid
                ),
            ),
            ('posted', lambda agent, system: system.time if agent.demand else agent.posted),
        )


class JobseekerView(AgentView):
    """Worker as the rules see it: its search and job states, and the parameters of separation and search.

    The states are the reservation wage `ask`, the search flag `supply`, the `employer`, the `wage`, the `duration` of
    search, and the steps `hired` and `offer` of the last hire. The parameters are the `separation` rate, the initial
    `reservation` wage, and its `decline` per step of search.
    """

    ask: float
    supply: float
    employer: float
    wage: float
    duration: float
    hired: float
    offer: float
    separation: float
    reservation: float
    decline: float


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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import JobseekerElement
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
        employer: Annotated[float, StateSpec(reference='firms')] = -1.0,
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

    @staticmethod
    def get_search() -> tuple[tuple[str, Callable[[JobseekerView, SystemView], float]], ...]:
        """Return the states a worker sets in the phase of offers before the separations, in their order.

        A worker unemployed at the start of the step searches, with a reservation wage that falls with the duration of
        its search.

        Returns:
            search:
                The name of every state and the function of its value, of the worker and the system view.
        """
        return (
            ('supply', lambda agent, system: float(agent.employer < 0)),
            (
                'ask',
                lambda agent, system: (
                    agent.reservation * (1 - agent.decline) ** agent.duration if agent.supply else agent.ask
                ),
            ),
        )

    @staticmethod
    def get_spell() -> tuple[tuple[str, Callable[[JobseekerView, SystemView], float]], ...]:
        """Return the states a worker sets in the phase of offers after the separations, in their order.

        A worker who lost its job earns no wage, and the duration of a search grows while the worker searches.

        Returns:
            spell:
                The name of every state and the function of its value, of the worker and the system view.
        """
        return (
            ('wage', lambda agent, system: 0.0 if agent.employer < 0 else agent.wage),
            (
                'duration',
                lambda agent, system: (
                    agent.duration + 1 if agent.supply else 0.0 if agent.employer < 0 else agent.duration
                ),
            ),
        )


def hire(trade: MatchSpec, system: SystemView) -> None:
    """Record the employer, the wage, and the steps of the hire and of its offer on the hired worker.

    The hired worker keeps its search flag `supply` until the next step, which the market lowered when it cleared, so
    the observable `searching` counts every worker who searched in the step.

    Args:
        trade:
            The match of a firm, the requester of labour, with a worker, its provider.

        system:
            The system view of the market.
    """
    worker, firm = trade.provider, trade.requester
    if not isinstance(worker, JobseekerView) or not isinstance(firm, EmployerView):
        return
    worker.employer, worker.wage, worker.duration = float(firm.id), trade.value, 0.0
    worker.supply += trade.quantity
    worker.hired, worker.offer = system.time, firm.posted


def _unemployment(employers: object) -> float:
    """Return the share of unemployed workers, whose employer is negative."""
    return float(np.mean(np.asarray(employers, dtype=float) < 0))


class LabourMarketSystem(BaseSystem):
    """Hamill and Gilbert's labour market of firms posting wage offers and unemployed workers searching.

    In the phase of offers, jobs are destroyed at the separation rate `s`, every firm posts a vacancy with a uniform
    wage offer at the vacancy rate, and unemployed workers set reservation wages that fall with the duration of their
    search. In the phase of matching, a deferred-acceptance market pairs vacancies with workers whose reservation wage
    they cover, every worker getting the best offer of a stable matching. The unemployment rate settles at the
    flow-balance value `s / (s + f)`, with `f` the share of searching workers hired in a step.

    The system has the groups `firms` and `workers`, the market `labour` with the reporters `labour.value` (the mean
    wage of the hires) and `labour.volume` (the hires), and the catalog observables of the unemployment rate
    `unemployment` and of the searching workers `searching`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import LabourMarketSystem
        >>> [group.size for group in LabourMarketSystem(20, 5).prepare().system_.elements]
        [5, 20]
    """

    field = 'economics/markets'
    topics = ('labour', 'market', 'matching', 'agents')
    references = ('Hamill and Gilbert 2016 ch. 7',)
    until = 200
    observables = (
        ObservableSpec('unemployment', group='workers', variable='employer', reduce=_unemployment),
        ObservableSpec('searching', group='workers', variable='supply', reduce='sum'),
    )

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_workers = n_workers
        self.n_firms = n_firms
        self.separation = separation
        self.vacancy_rate = vacancy_rate
        self.reservation = reservation
        self.decline = decline
        self.low_wage = low_wage
        self.high_wage = high_wage
        self.unemployment = unemployment
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        employed = rng.random(self.n_workers) >= self.unemployment
        employers = np.where(employed, rng.integers(self.n_firms, size=self.n_workers), -1).astype(float)
        firms = GroupElement(
            'firms',
            element=EmployerElement,
            size=self.n_firms,
            values={
                'vacancy_rate': self.vacancy_rate,
                'low_wage': self.low_wage,
                'high_wage': self.high_wage,
                'bid': self.low_wage,
            },
        )
        workers = GroupElement(
            'workers',
            element=JobseekerElement,
            size=self.n_workers,
            values={
                'employer': employers,
                'wage': np.where(employed, (self.low_wage + self.high_wage) / 2, 0.0),
                'supply': 0.0,
                'ask': self.reservation,
                'separation': self.separation,
                'reservation': self.reservation,
                'decline': self.decline,
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
            probability=lambda agent, system: agent.separation,
            form='agent',
            phase='offers',
        )
        spells = [
            MapInteraction(
                f'workers_{name}',
                target='workers',
                state=name,
                function=function,
                form='agent',
                phase='offers',
            )
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
            self.label,
            elements=[firms, workers],
            interactions=[
                *offers,
                losses,
                *spells,
                market,
            ],
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('offers'), PhaseSpec('matching')])),
        )
