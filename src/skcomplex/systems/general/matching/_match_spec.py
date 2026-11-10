"""Define the spec of one match of a requester with a provider."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.specs import BaseSpec


@dataclass(frozen=True)
class MatchSpec(BaseSpec):
    """A spec of one match, a quantity at a value per unit between two members.

    Unlike the rule specs, which say how members match, it records one match that a rule made. Choose it when an
    `on_match` function of a `MatchingInteraction` reads the match it receives.

    It appears in many fields. A buyer pays a seller for goods, a worker takes a job at a wage, and a borrower takes a
    loan at a rate.

    Args:
        requester:
            The agent view of the requester.

        provider:
            The agent view of the provider.

        quantity:
            The matched quantity.

        value:
            The clearing value per unit.

    Examples:
        >>> from skcomplex.systems.general.matching import MatchSpec
        >>> MatchSpec('a', 'b', 2.0, 5.0).quantity
        2.0
    """

    requester: object
    provider: object
    quantity: float
    value: float
