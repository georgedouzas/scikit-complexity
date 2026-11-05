"""Von Thünen land use, the crop of highest bid rent around one market."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.elements import FieldElement
from ....core.spaces import LatticeSpace, StaticTime
from ....core.system import System
from ..._catalog import register_model
from ._bid_rent import BidRentInteraction
from ._land_cell import LandCellElement


@register_model(
    field='economics/resources',
    tags=('land use', 'bid rent', 'static', 'lattice'),
    references=('Giordano 2024, arXiv 2407.02181', 'von Thünen 1826'),
)
def von_thunen(
    size: int = 41,
    *,
    yields: Sequence[float] = (4.0, 2.0, 1.0),
    prices: Sequence[float] = (3.0, 3.0, 3.5),
    costs: Sequence[float] = (1.0, 0.5, 0.5),
    rates: Sequence[float] = (0.2, 0.2, 0.2),
) -> System:
    """Return a square plain around one market at its centre, whose every cell grows the crop of highest bid rent.

    A crop of a high yield per cell bids most near the market, and its bid falls fastest with distance, so the land use
    forms concentric rings whose boundaries lie where the bids of neighbouring crops are equal (Giordano 2024 §7).

    Args:
        size:
            The side of the plain in cells, one unit of distance each.

        yields:
            The yield of every crop per cell.

        prices:
            The price of every crop at the market.

        costs:
            The cost of every crop per unit of yield.

        rates:
            The transport rate of every crop per unit of yield and of distance.

    Returns:
        system:
            The model, with the states `plain.crop` and `plain.distance`.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.economics.resources import von_thunen
        >>> trajectory = Simulation(von_thunen()).prepare().run(1.0).trajectory_
        >>> sorted(set(trajectory.values('plain.crop')[0].ravel().tolist()))
        [0.0, 1.0, 2.0, 3.0]
    """
    centre = (size - 1) / 2
    rows, columns = np.indices((size, size))
    distance = np.hypot(rows - centre, columns - centre)
    plain = FieldElement(
        'plain',
        element=LandCellElement,
        space=LatticeSpace((size, size)),
        values={'distance': distance},
    )
    bid = BidRentInteraction('bid', target='plain', yields=yields, prices=prices, costs=costs, rates=rates)
    return System(elements=[plain], interactions=[bid], time=StaticTime())
