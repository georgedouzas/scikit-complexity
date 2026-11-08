"""Markets systems: a hand car wash where passing cars queue for workers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class CarView(AgentView):
    """Car as the rules see it.

    It holds its `stage`, the `work` left on it, and the size of its `crew`.
    """

    stage: float
    work: float
    crew: float


WASH = 0


class CarElement(Element):
    """Car at the wash, waiting for or receiving a wash or a valet.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        stage:
            0 for the wash and 1 for the valet.

        work:
            The person-seconds of work left at the current stage.

        crew:
            The number of workers on the car in the current second.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import CarElement
        >>> GroupElement('cars', element=CarElement, size=1).size
        1
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stage: Annotated[float, StateSpec(domain=NN)] = WASH,
        work: Annotated[float, StateSpec()] = 525.0,
        crew: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stage = stage
        self.work = work
        self.crew = crew

    @classmethod
    def _get_view(cls) -> type[CarView]:
        """Return the view of a car, with its stage, work, and crew."""
        return CarView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the workers and the till act on the cars.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


VALET = 1


CREW_SIZE = 2


class WasherElement(Element):
    """Worker who joins the front car of the wash or of the valet.

    It joins while the car has fewer than two workers. A worker who finds both areas staffed helps at a random one of
    them. One second of work removes one person-second from the car.

    Args:
        label:
            The label of the agent or group.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import WasherElement
        >>> GroupElement('workers', element=WasherElement, size=4).size
        4
    """

    law_classes = (RuleLaw,)

    cars: ClassVar[str] = 'cars'

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the work rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.work)]

    def work(self, agent: AgentView, system: SystemView) -> None:
        """Work one second on the front car of an area.

        Args:
            agent:
                The worker.

            system:
                The system.
        """
        cars = system.agents(self.cars, CarView)
        fronts = [
            car for car in (WasherElement._front(cars, WASH), WasherElement._front(cars, VALET)) if car is not None
        ]
        if not fronts:
            return
        short = [car for car in fronts if car.crew < CREW_SIZE]
        car = short[0] if short else fronts[int(system.rng.integers(len(fronts)))]
        car.crew += 1
        car.work -= 1.0

    @staticmethod
    def _front(cars: list[CarView], stage: int) -> CarView | None:
        """Return the car that came first among the cars at a stage.

        It is `None` if there is none.
        """
        waiting = [car for car in cars if car.stage == stage]
        return min(waiting, key=lambda car: car.id) if waiting else None


class TillView(AgentView):
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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        revenue:
            The money taken.

        washes:
            The number of cars that left after a wash only.

        valets:
            The number of cars that left after a valet.

        turned_away:
            The number of drivers who did not turn in because of the queue.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import TillElement
        >>> GroupElement('till', element=TillElement, size=1).size
        1
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        revenue: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        washes: Annotated[float, StateSpec(domain=NN)] = 0,
        valets: Annotated[float, StateSpec(domain=NN)] = 0,
        turned_away: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.revenue = revenue
        self.washes = washes
        self.valets = valets
        self.turned_away = turned_away

    @classmethod
    def _get_view(cls) -> type[TillView]:
        """Return the view of the till, with its revenue and counts."""
        return TillView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the car wash books the payments.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


_PARAMETERS = ('p_pass', 'p_enter', 'p_valet', 'wash_work', 'valet_work', 'wash_price', 'valet_price', 'balk')


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

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        cars: Annotated[str, RoleSpec()],
        till: Annotated[str, RoleSpec()],
        p_pass: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.33,
        p_enter: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        p_valet: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        wash_work: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 525.0,
        valet_work: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1750.0,
        wash_price: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        valet_price: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        balk: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one second.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.operate)]

    def operate(self, system: SystemView) -> None:
        """Finish the cars whose work is done and reset the crews.

        Then a passing car turns in or drives on.

        Args:
            system:
                The system.
        """
        values = {name: float(system.parameter(name)) for name in _PARAMETERS}
        till = system.agents('till', TillView)[0]
        for car in system.agents('cars', CarView):
            car.crew = 0
            if car.work <= 0:
                self._finish(car, till, system, values)
        if system.rng.random() >= values['p_pass'] * values['p_enter']:
            return
        if sum(car.stage == WASH for car in system.agents('cars', CarView)) >= values['balk']:
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

    Examples:
        >>> from skcomplex.systems.economics.markets import CarWashSystem
        >>> CarWashSystem(5).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
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

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
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
        return System(
            self.label,
            elements=[
                GroupElement('workers', element=WasherElement, size=self.n_workers, variable=True),
                GroupElement(
                    'cars',
                    element=CarElement,
                    size=self.n_cars,
                    values={'work': self.wash_work},
                    variable=True,
                ),
                GroupElement('till', element=TillElement, size=1, variable=True),
            ],
            interactions=[operation],
        )
