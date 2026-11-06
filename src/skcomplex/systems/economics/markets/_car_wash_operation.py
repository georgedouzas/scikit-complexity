"""Arrival of passing cars, completion of washes and valets, and payment, once every second."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ._car import WASH
from ._car_view import CarView
from ._till_view import TillView
from ._washer import VALET

_PARAMETERS = ('p_pass', 'p_enter', 'p_valet', 'wash_work', 'valet_work', 'wash_price', 'valet_price', 'balk')


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
        >>> from skcomplex.systems.economics.markets import CarWashOperationInteraction
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

    def _finish(self, car: CarView, till: TillView, system: SystemView, values: dict[str, float]) -> None:
        """Move a car whose work is done to the valet, or take its payment and let it leave."""
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
        """Finish the cars whose work is done, reset the crews, and let a passing car turn in or drive on.

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
