"""Markets systems: a labour market of firms and unemployed workers.

Firms post wage offers and unemployed workers accept the best one.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Member
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import AgentView, BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.matching import MatchingInteraction, MatchSpec, StableMatchingSpec


class EmployerView(AgentView):
    """Employer as the rules see it.

    It holds its offer `bid`, vacancy `demand`, step `posted`, and its wage parameters. The parameters are the
    `vacancy_rate` and the range `low_wage` to `high_wage` of the wage offers.
    """

    bid: float
    demand: float
    posted: float
    vacancy_rate: float
    low_wage: float
    high_wage: float


class EmployerElement(Element):
    """Firm that posts a vacancy with a random wage offer in some steps.

    At most one worker fills the vacancy.

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

        size:
            The number of firms, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import EmployerElement
        >>> EmployerElement('firms', size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        bid: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        demand: Annotated[object, NN] = 0.0,
        posted: object = -1.0,
        vacancy_rate: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        low_wage: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        high_wage: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.bid = bid
        self.demand = demand
        self.posted = posted
        self.vacancy_rate = vacancy_rate
        self.low_wage = low_wage
        self.high_wage = high_wage

    @classmethod
    def _get_view(cls) -> type[EmployerView]:
        """Return the view of an employer.

        It holds its vacancy and the parameters of its offers.
        """
        return EmployerView


class JobseekerView(AgentView):
    """Worker as the rules see it.

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
    """Worker who loses its job at the separation rate.

    While unemployed, it lowers its reservation wage. A worker who loses its job searches from the next step. An
    unemployed worker asks the reservation wage `w_0 (1 - decline)^d`, where `d` is the number of steps of search, and
    is hired by the matching market.

    Args:
        label:
            The label of the agent or group.

        ask:
            The reservation wage.

        supply:
            One while the worker searches in the current step, zero otherwise.

        employer:
            The index of the employer among the firms, or -1 for an unemployed worker.

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

        size:
            The number of workers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import JobseekerElement
        >>> JobseekerElement('workers', size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        ask: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        supply: Annotated[object, NN] = 1.0,
        employer: Annotated[object, Member('firms')] = -1,
        wage: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        duration: Annotated[object, NN] = 0.0,
        hired: object = -1.0,
        offer: object = -1.0,
        separation: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        reservation: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        decline: Annotated[object, RealSet.closed_open(0, oo)] = 0.1,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
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
        """Return the view of a worker.

        It holds its search and job states and their parameters.
        """
        return JobseekerView


class JobOffersInteraction(Interaction):
    """Phase of offers of the labour market, before the firms meet the workers.

    Every firm withdraws its last vacancy and posts a new one at the vacancy rate, with a wage offer drawn uniformly
    from `low_wage` to `high_wage` and the step of posting. Every worker unemployed at the start of the step searches,
    with a reservation wage `reservation (1 - decline)^d` that falls with the duration `d` of its search. Every employed
    worker then loses its job at the separation rate, a worker without a job earns no wage, and the duration of a
    search grows while the worker searches.

    Args:
        label:
            The label of the interaction.

        firms:
            The path of the group of firms.

        workers:
            The path of the group of workers.

        phase:
            The phase the offers run in.

    Examples:
        >>> from skcomplex.systems.economics.markets import JobOffersInteraction
        >>> JobOffersInteraction('offers', firms='firms', workers='workers').phase
        'offers'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, firms: Role, workers: Role, phase: str | None = 'offers') -> None:
        super().__init__(label)
        self.firms = firms
        self.workers = workers
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the offers of every step as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self.offer, phase=self.phase)]

    def offer(self, system: SystemView) -> None:
        """Post the vacancies of the firms, then set the search, the separations, and the spells of the workers.

        Every step of the phase runs for every firm or worker before the next one.

        Args:
            system:
                The system view of the offers.
        """
        firms = system.get_agents('firms', EmployerView)
        for firm in firms:
            firm.demand = float(system.rng.random() < firm.vacancy_rate)
        for firm in firms:
            firm.bid = float(system.rng.uniform(firm.low_wage, firm.high_wage)) if firm.demand else firm.bid
        for firm in firms:
            firm.posted = float(system.time if firm.demand else firm.posted)
        workers = system.get_agents('workers', JobseekerView)
        for worker in workers:
            worker.supply = float(worker.employer < 0)
        for worker in workers:
            worker.ask = worker.reservation * (1 - worker.decline) ** worker.duration if worker.supply else worker.ask
        for worker in workers:
            if worker.employer >= 0 and system.rng.random() < worker.separation:
                worker.employer = -1.0
        for worker in workers:
            worker.wage = 0.0 if worker.employer < 0 else worker.wage
        for worker in workers:
            searching = worker.duration + 1 if worker.supply else 0.0 if worker.employer < 0 else worker.duration
            worker.duration = float(searching)


def hire(trade: MatchSpec, system: SystemView) -> None:
    """Record the employer, the wage, and the steps of the hire on the hired worker.

    The steps are those of the hire and of its offer. The hired worker keeps its search flag `supply` until the next
    step, which the market lowered when it cleared, so the observable `searching` counts every worker who searched in
    the step.

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


def _get_unemployment(employers: object) -> float:
    """Return the share of unemployed workers, whose employer is negative."""
    return float(np.mean(np.asarray(employers, dtype=float) < 0))


class LabourMarketSystem(BaseSystem):
    """Hamill and Gilbert's labour market.

    Firms post wage offers and unemployed workers search. In the phase of offers, jobs are destroyed at the separation
    rate `s`, every firm posts a vacancy with a uniform wage offer at the vacancy rate, and unemployed workers set
    reservation wages that fall with the duration of their search. In the phase of matching, a deferred-acceptance
    market pairs vacancies with workers whose reservation wage they cover, every worker getting the best offer of a
    stable matching. The unemployment rate settles at the flow-balance value `s / (s + f)`, with `f` the share of
    searching workers hired in a step.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        Observable('unemployment', group='workers', variable='employer', reduce=_get_unemployment),
        Observable('searching', group='workers', variable='supply', reduce='sum'),
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        employed = rng.random(self.n_workers) >= self.unemployment
        employers = np.where(employed, rng.integers(self.n_firms, size=self.n_workers), -1)
        firms = EmployerElement(
            'firms',
            vacancy_rate=self.vacancy_rate,
            low_wage=self.low_wage,
            high_wage=self.high_wage,
            bid=self.low_wage,
            size=self.n_firms,
        )
        workers = JobseekerElement(
            'workers',
            employer=employers,
            wage=np.where(employed, (self.low_wage + self.high_wage) / 2, 0.0),
            supply=0.0,
            ask=self.reservation,
            separation=self.separation,
            reservation=self.reservation,
            decline=self.decline,
            size=self.n_workers,
        )
        market = MatchingInteraction(
            'labour',
            requesters='firms',
            providers='workers',
            rule=StableMatchingSpec(value='requester'),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            phase='matching',
            on_match=hire,
        )
        self.system_ = System(
            self.label,
            elements=[firms, workers],
            interactions=[
                JobOffersInteraction('offers', firms='firms', workers='workers'),
                market,
            ],
            solver=MesaSolver(phases=[PhaseSpec('offers'), PhaseSpec('matching')]),
        )
