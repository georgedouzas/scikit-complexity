"""Link of a supply chain, which ships beer from the upstream echelon to the downstream echelon."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import TransferInteraction
from ....core.laws.utils import distributed_delay
from ....specs import RoleSpec
from ._echelon import EchelonElement


class SupplyLinkInteraction(TransferInteraction):
    """Link of a supply chain, which ships beer from the upstream echelon to the downstream echelon.

    The shipments and their arrivals, after a third-order delay of the shipping delay of the downstream echelon, are
    the legs of one transfer, from the inventory of the upstream echelon to the beer in transit to the downstream
    echelon and on to its inventory.

    Args:
        label:
            The label of the link.

        downstream:
            The path of the echelon that orders.

        upstream:
            The path of the echelon that ships.

    Examples:
        >>> from skcomplex.models.economics.markets import SupplyLinkInteraction
        >>> SupplyLinkInteraction('link', downstream='retailer', upstream='wholesaler').upstream
        'wholesaler'
    """

    def __init__(
        self,
        label: str,
        *,
        downstream: Annotated[str, RoleSpec()],
        upstream: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=upstream,
            target=downstream,
            state='inventory',
            into='in_transit',
            amount=lambda s: EchelonElement._shipments(s.upstream),
            flow='shipments',
            legs=[
                (
                    'downstream.in_transit',
                    'downstream.inventory',
                    lambda s: SupplyLinkInteraction._arrivals(s.upstream, s.downstream),
                    'arrivals',
                ),
            ],
        )
        self.downstream = downstream
        self.upstream = upstream

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the upstream echelon, which ships, and the downstream echelon, which receives."""
        return [(namespace.upstream, namespace.downstream)]

    @staticmethod
    def _arrivals(upstream: SimpleNamespace, downstream: SimpleNamespace) -> Expression:
        """Return the arrivals at the downstream echelon, the shipments after a third-order shipping delay."""
        return distributed_delay(EchelonElement._shipments(upstream), order=3, time=downstream.shipping_delay)
