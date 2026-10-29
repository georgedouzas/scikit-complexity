"""Markets models: a hand car wash where passing cars queue for workers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model

_PARAMETERS = ('p_pass', 'p_enter', 'p_valet', 'wash_work', 'valet_work', 'wash_price', 'valet_price', 'balk')


WASH = 0
VALET = 1
CREW_SIZE = 2


def _front(cars: list[AgentView], stage: int) -> AgentView | None:
    """Return the car that came first among the cars at a stage, or `None` if there is none."""
    waiting = [car for car in cars if car.stage == stage]
    return min(waiting, key=lambda car: car.id) if waiting else None


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

    Examples:
        >>> from skcomplex.models.economics.markets import CarElement
        >>> GroupElement('cars', element=CarElement, size=1).size
        1
    """

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


class WasherElement(Element):
    """Worker who joins the front car of the wash or of the valet while it has fewer than two workers.

    A worker who finds both areas staffed helps at a random one of them. One second of work removes one
    person-second from the car.

    Args:
        label:
            The label of the agent or group.

    Examples:
        >>> from skcomplex.models.economics.markets import WasherElement
        >>> GroupElement('workers', element=WasherElement, size=4).size
        4
    """

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

    def work(self, agent: AgentView, model: ModelView) -> None:
        """Work one second on the front car of an area.

        Args:
            agent:
                The worker.

            model:
                The model.
        """
        cars = model.agents(self.cars, AgentView)
        fronts = [car for car in (_front(cars, WASH), _front(cars, VALET)) if car is not None]
        if not fronts:
            return
        short = [car for car in fronts if car.crew < CREW_SIZE]
        car = short[0] if short else fronts[int(model.rng.integers(len(fronts)))]
        car.crew += 1
        car.work -= 1.0


class TillElement(Element):
    """Till of the car wash, which counts the revenue and the cars served and turned away.

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

    Examples:
        >>> from skcomplex.models.economics.markets import TillElement
        >>> GroupElement('till', element=TillElement, size=1).size
        1
    """

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


class CarWashOperationInteraction(Interaction):
    """Arrival of passing cars, completion of washes and valets, and payment, once every second.

    A passing car turns in with a probability, unless the wash queue holds `balk` cars or more. A washed car stays for
    a valet with a probability, and pays the wash price or the valet price when it leaves.

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
        >>> from skcomplex.models.economics.markets import CarWashOperationInteraction
        >>> CarWashOperationInteraction('wash', cars='cars', till='till', p_enter=0.01).p_enter
        0.01
    """

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

    def _finish(self, car: AgentView, till: AgentView, model: ModelView, values: dict[str, float]) -> None:
        """Move a car whose work is done to the valet, or take its payment and let it leave."""
        if car.stage == WASH and model.rng.random() < values['p_valet']:
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

    def operate(self, model: ModelView) -> None:
        """Finish the cars whose work is done, reset the crews, and let a passing car turn in or drive on.

        Args:
            model:
                The model.
        """
        values = {name: float(model.parameter(name)) for name in _PARAMETERS}
        till = model.agents('till', AgentView)[0]
        for car in model.agents('cars', AgentView):
            car.crew = 0
            if car.work <= 0:
                self._finish(car, till, model, values)
        if model.rng.random() >= values['p_pass'] * values['p_enter']:
            return
        if sum(car.stage == WASH for car in model.agents('cars', AgentView)) >= values['balk']:
            till.turned_away += 1
            return
        model.create('cars', stage=WASH, work=values['wash_work'])


@register_model(
    field='economics/markets',
    tags=('queue', 'agents', 'labour'),
    references=('Chesney 2021',),
    defaults={'until': 600},
)
def car_wash(
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
) -> System:
    """Return Chesney's Clean and Wax hand car wash, a shadow account of the revenue of its workers' labour.

    Every step is one second. Passing cars turn in with a probability and queue for the wash, washed cars may queue
    for a valet, and the workers serve the front car of each area, two at a time if they can. The wash opens with
    cars already queuing.

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

    Returns:
        system:
            The model, with the groups `workers`, `cars`, and `till`.

    Examples:
        >>> from skcomplex.models.economics.markets import car_wash
        >>> car_wash(5).elements[0].check().prepare().size_
        5
    """
    operation = CarWashOperationInteraction(
        'wash',
        cars='cars',
        till='till',
        p_pass=p_pass,
        p_enter=p_enter,
        p_valet=p_valet,
        wash_work=wash_work,
        valet_work=valet_work,
        wash_price=wash_price,
        valet_price=valet_price,
        balk=float(balk),
    )
    return System(
        elements=[
            GroupElement('workers', element=WasherElement, size=n_workers, variable=True),
            GroupElement('cars', element=CarElement, size=n_cars, values={'work': wash_work}, variable=True),
            GroupElement('till', element=TillElement, size=1, variable=True),
        ],
        interactions=[operation],
    )
