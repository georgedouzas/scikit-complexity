"""Queues: birth-death processes of the number of customers waiting at a desk."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import BaseLaw, NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import JumpLaw, Symbols, where
from ....core.system import System
from ....engine.runtime import register_model
from ...core.elements import GroupElement


class QueueElement(Element):
    """Queue at a desk, a birth-death process whose length grows by arrivals and shrinks by departures.

    Customers arrive at rate `arrival_rate` and leave at rate `service_rate` while the queue is not empty.

    Args:
        label:
            The label of the queue or the group.

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

    n: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE_INTEGER)
    arrival_rate: float = parameter(default=0.5, domain=NumericDomain.NONNEGATIVE)
    service_rate: float = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the arrivals and the departures.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            JumpLaw(s.arrival_rate, {s.n: 1}),
            JumpLaw(where(s.n > 0, s.service_rate, 0), {s.n: -1}),
        ]


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
        >>> birth_death_queue(5).elements[0].size
        5
    """
    queue = GroupElement(
        'queues',
        element=QueueElement,
        size=n_queues,
        values={'n': n, 'arrival_rate': arrival_rate, 'service_rate': service_rate},
    )
    return System(elements=[queue])
