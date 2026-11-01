"""Markets models: a hand car wash where passing cars queue for workers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import GroupElement
from ....core.system import System
from ..._catalog import register_model
from ._car import CarElement
from ._car_wash_operation import CarWashOperationInteraction
from ._till import TillElement
from ._washer import WasherElement


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
