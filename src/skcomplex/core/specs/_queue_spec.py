"""Define the queue spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

ORDERS = ('random', 'limit', 'sequential')


@dataclass(frozen=True)
class QueueSpec(BaseSpec):
    """Queue of requesters served by the lowest providers until the quantities run out.

    The last requesters are therefore rationed. Requesters queue in random order, by decreasing limit, or in order of
    creation, and each matches at the limit of the lowest providers whose limit it accepts.

    Args:
        order:
            The order of the queue, `'random'`, `'limit'`, or `'sequential'`.

    Raises:
        ValidationError:
            The order is unknown, raised when its owner checks it.

    Examples:
        >>> from skcomplex.core.specs import QueueSpec
        >>> QueueSpec(order='limit').order
        'limit'
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
