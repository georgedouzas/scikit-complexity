"""Natural resources systems: harvesting renewable stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ._farmer import FarmerElement
from ._harvest import HarvestInteraction
from ._stock import StockElement


@register_system(
    field='economics/resources',
    tags=('socio-ecological', 'agents', 'ode', 'co-simulation'),
    references=('Hamill and Gilbert 2016 ch. 10',),
    defaults={'until': 50},
)
def harvesting(farmers: int = 10, *, effort: float = 0.05, r: float = 1.0, K: float = 1000.0) -> System:
    """Return a socio-ecological system of farmer agents harvesting a logistically regrowing resource.

    The resource settles at `K * (1 - effort * farmers / r)` when `effort * farmers < r`, and collapses otherwise.

    Args:
        farmers:
            The number of farmers.

        effort:
            The harvest rate per farmer and unit of stock.

        r:
            The growth rate of the resource.

        K:
            The carrying capacity of the resource.

    Returns:
        system:
            The system, with the ODE subsystem `land` and the agent subsystem `village`.

    Examples:
        >>> from skcomplex.systems.economics.resources import harvesting
        >>> [subsystem.label for subsystem in harvesting().systems]
        ['land', 'village']
    """
    land = System('land', elements=[StockElement('stock', n=K, r=r, K=K)])
    village = System('village', elements=[GroupElement('farmers', element=FarmerElement, size=farmers)])
    harvest = HarvestInteraction('harvest', target='land/stock', harvesters='village/farmers', effort=effort)
    return System(interactions=[harvest], systems=[land, village])
