"""Harvest of a stock proportional to the stock and to the number of harvesters, the rate `-effort * N * n`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import count
from ....core.specs import ParameterSpec, RoleSpec


class HarvestInteraction(RateInteraction):
    """Harvest of a stock proportional to the stock and to the number of harvesters, the rate `-effort * N * n`.

    A rate on the state `n` of the target whose coefficient counts the members of the group of harvesters, which may
    live in another subsystem.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the harvested stock.

        harvesters:
            The path of the group of harvesters, which the harvest counts.

        effort:
            The harvest rate per harvester and unit of stock.

        state:
            The name of the state of the target that holds the stock.

    Examples:
        >>> from skcomplex.systems.economics.resources import HarvestInteraction
        >>> HarvestInteraction('harvest', target='land/stock', harvesters='village/farmers', effort=0.05).effort
        0.05
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        harvesters: Annotated[str, RoleSpec()],
        effort: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.harvesters = harvesters
        self.effort = effort

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the harvest `-effort * N * n`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the target and the harvesters.

        Returns:
            rate:
                The harvest rate of the stock.
        """
        return -namespace.effort * count(namespace.harvesters) * getattr(namespace.target, self.state)
