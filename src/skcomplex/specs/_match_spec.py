"""Define the match spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class MatchSpec(BaseSpec):
    """Match of a requester with a provider, of a quantity at a clearing value per unit.

    Args:
        requester:
            The agent view of the requester.

        provider:
            The agent view of the provider.

        quantity:
            The matched quantity.

        price:
            The clearing value per unit.

    Examples:
        >>> from skcomplex.specs import MatchSpec
        >>> MatchSpec('a', 'b', 2.0, 5.0).check().quantity
        2.0
    """

    requester: object
    provider: object
    quantity: float
    price: float
