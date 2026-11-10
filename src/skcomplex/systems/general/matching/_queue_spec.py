"""Define the spec of a queue of requesters served by the cheapest providers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.errors import ValidationError
from ....core.specs import BaseSpec

ORDERS = ('random', 'reserve', 'sequential')


@dataclass(frozen=True)
class QueueSpec(BaseSpec):
    """A spec of a queue of requesters served by the cheapest providers.

    Requesters queue in random order, by decreasing reserve, or in order of creation, and each matches at the reserve of
    the cheapest providers whose reserve it accepts. The last requesters may get nothing.

    Unlike `IterativeMatchingSpec`, which finds one value for all, it serves requesters one by one at the prices of the
    providers. Choose it when members are served in order, first come first served.

    It appears in many fields. Customers queue at a shop, patients wait for organs, and households queue for housing.

    Args:
        order:
            The order of the queue, `'random'`, `'reserve'`, or `'sequential'`.

    Raises:
        ValidationError:
            The order is unknown, raised when its owner checks it.

    Examples:
        >>> from skcomplex.systems.general.matching import QueueSpec
        >>> QueueSpec(order='reserve').order
        'reserve'
    """

    order: str = 'random'

    def _check(self) -> None:
        """Raise an error for an unknown order of the queue.

        Raises:
            ValidationError:
                The order is unknown.
        """
        if self.order not in ORDERS:
            error_msg = f'Unknown order {self.order!r}. Use one of {ORDERS}.'
            raise ValidationError(error_msg)
