"""Queues: birth-death processes of the number of customers waiting at a desk."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import JumpInteraction
from ....core.laws.utils import where
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class QueueElement(Element):
    """Queue at a desk, the number of customers waiting, with the rates of their arrivals and departures.

    The jumps of the model read the rates: customers arrive at rate `arrival_rate` and leave at rate `service_rate`
    while the queue is not empty.

    Args:
        label:
            The label of the queue or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of customers.

        arrival_rate:
            The rate of arrivals.

        service_rate:
            The rate of departures from a queue that is not empty.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import QueueElement
        >>> QueueElement('desk', arrival_rate=0.8).arrival_rate
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 0.0,
        arrival_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        service_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate


@register_model(
    field='mathematics/stochastic_processes',
    tags=('jump process', 'birth-death process', 'queueing'),
    references=('Bertin 2021', 'Kendall 1953', 'Feller 1968'),
    defaults={'until': 200.0, 'dt': 0.5},
)
def birth_death_queue(
    n_queues: int = 100,
    *,
    arrival_rate: float = 0.5,
    service_rate: float = 1.0,
    n: float = 0.0,
) -> System:
    """Return independent M/M/1 queues, birth-death processes with constant arrival and service rates.

    Customers arrive as a Poisson process of rate `lambda` and are served one at a time in exponential times of
    rate `mu`. For the load `rho = lambda / mu < 1` the length of an unbounded queue reaches the stationary
    geometric distribution `(1 - rho) rho^n`, of mean `rho / (1 - rho)` and variance `rho / (1 - rho)^2`, and for
    `rho > 1` it grows without bound.

    Args:
        n_queues:
            The number of independent queues.

        arrival_rate:
            The rate `lambda` of arrivals.

        service_rate:
            The rate `mu` of departures.

        n:
            The initial number of customers of every queue.

    Returns:
        system:
            The model, with the lengths `queues.n`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import birth_death_queue
        >>> birth_death_queue(5).elements[0].prepare().size_
        5
    """
    queue = GroupElement(
        'queues',
        element=QueueElement,
        size=n_queues,
        values={'n': n, 'arrival_rate': arrival_rate, 'service_rate': service_rate},
    )
    arrivals = JumpInteraction('arrivals', target='queues', changes={'n': 1}, rate=lambda s: s.target.arrival_rate)
    departures = JumpInteraction(
        'departures',
        target='queues',
        changes={'n': -1},
        rate=lambda s: where(s.target.n > 0, s.target.service_rate, 0),
    )
    return System(elements=[queue], interactions=[arrivals, departures])
