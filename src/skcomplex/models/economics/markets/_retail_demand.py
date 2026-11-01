"""Customers of a supply chain, who take the shipments of the retailer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import TransferInteraction
from ....core.laws import Sink
from ....specs import RoleSpec
from ._echelon import EchelonElement


class RetailDemandInteraction(TransferInteraction):
    """Customers of a supply chain, who take the shipments of the retailer.

    The sales are a transfer from the inventory of the retailer to the customers.

    Args:
        label:
            The label of the customers.

        retailer:
            The path of the retailer.

    Examples:
        >>> from skcomplex.models.economics.markets import RetailDemandInteraction
        >>> RetailDemandInteraction('customers', retailer='retailer').retailer
        'retailer'
    """

    def __init__(
        self,
        label: str,
        *,
        retailer: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=retailer,
            target=retailer,
            state='inventory',
            into=Sink('customers'),
            amount=lambda s: EchelonElement._shipments(s.retailer),
            flow='sales',
        )
        self.retailer = retailer

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the retailer, whose inventory the customers take."""
        return [(namespace.retailer, namespace.retailer)]
