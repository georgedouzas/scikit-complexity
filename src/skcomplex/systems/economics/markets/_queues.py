"""Markets systems: a hand car wash where passing cars queue for workers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role

WASH = 0
VALET = 1
CREW_SIZE = 2
PARAMETERS = ('p_pass', 'p_enter', 'p_valet', 'wash_work', 'valet_work', 'wash_price', 'valet_price', 'balk')


class CarView(AgentViewProtocol, Protocol):
    """Car as the rules see it.

    It holds its `stage`, the `work` left on it, and the size of its `crew`.
    """

    stage: float
    work: float
    crew: float


class CarElement(Element):
    """Car at the wash, waiting for or receiving a wash or a valet.

    Args:
        label:
            The label of the agent or group.

        stage:
            0 for the wash and 1 for the valet.

        work:
            The person-seconds of work left at the current stage.

        crew:
            The number of workers on the car in the current second.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import CarElement
        >>> CarElement('cars', size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        *,
        stage: Annotated[object, NN] = WASH,
        work: object = 525.0,
        crew: Annotated[object, NN] = 0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.stage = stage
        self.work = work
        self.crew = crew


class WasherElement(Element):
    """Worker of the car wash, whose work `WashingInteraction` sets.

    Args:
        label:
            The label of the agent or group.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import WasherElement
        >>> WasherElement('workers', size=4).size
        4
    """

    def __init__(self, label: str, *, size: int | None = None) -> None:
        super().__init__(label, size=size)


class WashingInteraction(Interaction):
    """Work of the workers of the car wash on the front car of the wash or of the valet.

    A worker joins the front car while it has fewer than two workers. A worker who finds both areas staffed helps at a
    random one of them. One second of work removes one person-second from the car.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of workers.

        cars:
            The path of the group of cars.

    Examples:
        >>> from skcomplex.systems.economics.markets import WashingInteraction
        >>> WashingInteraction('washing', target='workers', cars='cars').cars
        'cars'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role, cars: Role) -> None:
        super().__init__(label)
        self.target = target
        self.cars = cars

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the work of every worker as its rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.work)]

    def work(self, agent: AgentViewProtocol, system: SystemView) -> None:
        """Work one second on the front car of an area.

        Args:
            agent:
                The worker.

            system:
                The system.
        """
        cars = cast(list[CarView], system.get_agents('cars'))
        fronts = [
            car
            for car in (WashingInteraction._get_front(cars, WASH), WashingInteraction._get_front(cars, VALET))
            if car is not None
        ]
        if not fronts:
            return
        short = [car for car in fronts if car.crew < CREW_SIZE]
        car = short[0] if short else fronts[int(system.rng.integers(len(fronts)))]
        car.crew += 1
        car.work -= 1.0

    @staticmethod
    def _get_front(cars: list[CarView], stage: int) -> CarView | None:
        """Return the car that came first among the cars at a stage.

        It is `None` if there is none.
        """
        waiting = [car for car in cars if car.stage == stage]
        return min(waiting, key=lambda car: car.id) if waiting else None


class TillView(AgentViewProtocol, Protocol):
    """Till as the operation sees it.

    It holds its `revenue` and its counts of `washes`, `valets`, and cars `turned_away`.
    """

    revenue: float
    washes: float
    valets: float
    turned_away: float


class TillElement(Element):
    """Till of the car wash.

    It counts the revenue and the cars served and turned away.

    Args:
        label:
            The label of the agent or group.

        revenue:
            The money taken.

        washes:
            The number of cars that left after a wash only.

        valets:
            The number of cars that left after a valet.

        turned_away:
            The number of drivers who did not turn in because of the queue.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import TillElement
        >>> TillElement('till', size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        *,
        revenue: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        washes: Annotated[object, NN] = 0,
        valets: Annotated[object, NN] = 0,
        turned_away: Annotated[object, NN] = 0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.revenue = revenue
        self.washes = washes
        self.valets = valets
        self.turned_away = turned_away


class CarWashOperationInteraction(Interaction):
    """Arrival of passing cars, completion of washes and valets, and payment.

    They happen once every second. A passing car turns in with a probability, unless the wash queue holds `balk` cars or
    more. A washed car stays for a valet with a probability, and pays the wash price or the valet price when it leaves.

    Args:
        label:
            The label of the interaction.

        cars:
            The path of the group of cars.

        till:
            The path of the group that holds the till.

        p_pass:
            The probability that a car passes in a second.

        p_enter:
            The probability that a passing car turns in.

        p_valet:
            The probability that a washed car stays for a valet.

        wash_work:
            The person-seconds of a wash.

        valet_work:
            The person-seconds of a valet.

        wash_price:
            The price of a wash.

        valet_price:
            The price of a wash with a valet.

        balk:
            The length of the wash queue at which drivers no longer turn in.

    Examples:
        >>> from skcomplex.systems.economics.markets import CarWashOperationInteraction
        >>> CarWashOperationInteraction('wash', cars='cars', till='till', p_enter=0.01).p_enter
        0.01
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        cars: Role,
        till: Role,
        p_pass: Annotated[float, RealSet.closed_open(0, oo)] = 0.33,
        p_enter: Annotated[float, RealSet.closed_open(0, oo)] = 0.005,
        p_valet: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        wash_work: Annotated[float, RealSet.open(0, oo)] = 525.0,
        valet_work: Annotated[float, RealSet.open(0, oo)] = 1750.0,
        wash_price: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        valet_price: Annotated[float, RealSet.closed_open(0, oo)] = 10.0,
        balk: Annotated[float, RealSet.open(0, oo)] = 3.0,
    ) -> None:
        super().__init__(label)
        self.cars = cars
        self.till = till
        self.p_pass = p_pass
        self.p_enter = p_enter
        self.p_valet = p_valet
        self.wash_work = wash_work
        self.valet_work = valet_work
        self.wash_price = wash_price
        self.valet_price = valet_price
        self.balk = balk

    def _finish(self, car: CarView, till: TillView, system: SystemView, values: dict[str, float]) -> None:
        """Move a car whose work is done to the valet.

        Otherwise, take its payment and let it leave.
        """
        if car.stage == WASH and system.rng.random() < values['p_valet']:
            car.stage, car.work = VALET, values['valet_work']
            return
        if car.stage == WASH:
            till.revenue += values['wash_price']
            till.washes += 1
        else:
            till.revenue += values['valet_price']
            till.valets += 1
        car.remove()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule of one second as the rule of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self.operate)]

    def operate(self, system: SystemView) -> None:
        """Finish the cars whose work is done and reset the crews.

        Then a passing car turns in or drives on.

        Args:
            system:
                The system.
        """
        values = {name: float(system.get_parameter(name)) for name in PARAMETERS}
        till = cast(list[TillView], system.get_agents('till'))[0]
        for car in cast(list[CarView], system.get_agents('cars')):
            car.crew = 0
            if car.work <= 0:
                self._finish(car, till, system, values)
        if system.rng.random() >= values['p_pass'] * values['p_enter']:
            return
        if sum(car.stage == WASH for car in cast(list[CarView], system.get_agents('cars'))) >= values['balk']:
            till.turned_away += 1
            return
        system.create('cars', stage=WASH, work=values['wash_work'])


class CarWashSystem(BaseSystem):
    """Chesney's Clean and Wax hand car wash.

    It is a shadow account of the revenue of its workers' labour. Every step is one second. Passing cars turn in with a
    probability and queue for the wash, washed cars may queue for a valet, and the workers serve the front car of each
    area, two at a time if they can. The wash opens with cars already queuing. The system has the groups `workers`,
    `cars`, and `till`.

    Args:
        n_workers:
            The number of workers.

        n_cars:
            The number of cars queuing for the wash at the start, at least one.

        p_pass:
            The probability that a car passes in a second.

        p_enter:
            The probability that a passing car turns in.

        p_valet:
            The probability that a washed car stays for a valet.

        wash_work:
            The person-seconds of a wash.

        valet_work:
            The person-seconds of a valet.

        wash_price:
            The price of a wash.

        valet_price:
            The price of a wash with a valet.

        balk:
            The length of the wash queue at which drivers no longer turn in.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import CarWashSystem
        >>> CarWashSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'economics/markets'
    topics = ('queue', 'agents', 'labour')
    references = ('Chesney 2021',)
    until = 600

    def __init__(
        self,
        n_workers: int = 4,
        n_cars: int = 1,
        *,
        p_pass: float = 0.33,
        p_enter: float = 0.005,
        p_valet: float = 0.5,
        wash_work: float = 525.0,
        valet_work: float = 1750.0,
        wash_price: float = 4.0,
        valet_price: float = 10.0,
        balk: int = 3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_workers = n_workers
        self.n_cars = n_cars
        self.p_pass = p_pass
        self.p_enter = p_enter
        self.p_valet = p_valet
        self.wash_work = wash_work
        self.valet_work = valet_work
        self.wash_price = wash_price
        self.valet_price = valet_price
        self.balk = balk

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        operation = CarWashOperationInteraction(
            'wash',
            cars='cars',
            till='till',
            p_pass=self.p_pass,
            p_enter=self.p_enter,
            p_valet=self.p_valet,
            wash_work=self.wash_work,
            valet_work=self.valet_work,
            wash_price=self.wash_price,
            valet_price=self.valet_price,
            balk=float(self.balk),
        )
        self.system_ = System(
            self.label,
            elements=[
                WasherElement('workers', size=self.n_workers),
                CarElement('cars', work=self.wash_work, size=self.n_cars),
                TillElement('till', size=1),
            ],
            interactions=[WashingInteraction('washing', target='workers', cars='cars'), operation],
        )
