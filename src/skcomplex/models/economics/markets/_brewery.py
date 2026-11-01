"""Brewery of a supply chain, which brews the orders of the factory over its production delay."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import TransferInteraction
from ....core.laws import Source
from ....core.laws.utils import distributed_delay, maximum
from ....specs import RoleSpec


class BreweryInteraction(TransferInteraction):
    """Brewery of a supply chain, which brews the orders of the factory over its production delay.

    The brewing is a transfer from the brewery into the beer in production at the factory and on to its inventory.

    Args:
        label:
            The label of the brewery.

        factory:
            The path of the factory.

    Examples:
        >>> from skcomplex.models.economics.markets import BreweryInteraction
        >>> BreweryInteraction('brewery', factory='factory').factory
        'factory'
    """

    def __init__(
        self,
        label: str,
        *,
        factory: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=factory,
            target=factory,
            state=Source('brewery'),
            into='in_transit',
            amount=lambda s: BreweryInteraction._orders(s.factory),
            flow='starts',
            legs=[('in_transit', 'inventory', lambda s: BreweryInteraction._completions(s.factory), 'completions')],
        )
        self.factory = factory

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the factory, whose beer in production and inventory the brewery fills."""
        return [(namespace.factory, namespace.factory)]

    @staticmethod
    def _orders(echelon: SimpleNamespace) -> Expression:
        """Return the orders of an echelon by Sterman's anchoring and adjustment rule, never negative."""
        effective = echelon.inventory - echelon.backlog
        orders = echelon.expected + echelon.alpha * (echelon.desired - effective - echelon.beta * echelon.supply_line)
        return maximum(orders, 0)

    @staticmethod
    def _completions(factory: SimpleNamespace) -> Expression:
        """Return the brews completed for the factory, its orders after a third-order production delay."""
        return distributed_delay(BreweryInteraction._orders(factory), order=3, time=factory.production_delay)
