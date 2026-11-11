"""Von Thünen land use, the crop of highest bid rent around one market."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....core.components.base import BaseComponent
from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import Interaction
from ....core.components.laws import EquationLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace, LatticeSpace, StaticTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import RoleSpec, StateSpec


class LandCellElement(Element):
    """Cell of a plain that grows one crop, numbered from 1, or lies idle as crop 0.

    Args:
        label:
            The label of the cell or the field.

        crop:
            The initial guess of the crop, which the equation replaces.

        distance:
            The distance from the market.

    Examples:
        >>> from skcomplex.systems.economics.resources import LandCellElement
        >>> LandCellElement('cell', distance=3.0).distance
        3.0
    """

    def __init__(
        self,
        label: str,
        crop: Annotated[float, StateSpec()] = 0.0,
        distance: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.crop = crop
        self.distance = distance


class BidRentInteraction(Interaction):
    """Land use of highest bid rent.

    Crop `i` bids `Y_i (p_i - c_i) - Y_i f_i d` at the distance `d` of a market. A cell grows the crop of the highest
    positive bid, and lies idle as crop 0 when no bid is positive.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the field of cells, whose members hold `crop` and `distance`.

        yields:
            The yield `Y_i` of every crop per cell.

        prices:
            The price `p_i` of every crop at the market.

        costs:
            The cost `c_i` of every crop per unit of yield.

        rates:
            The transport rate `f_i` of every crop per unit of yield and of distance.

    Raises:
        ValidationError:
            The four sequences do not have one value per crop, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.economics.resources import BidRentInteraction
        >>> bid = BidRentInteraction('bid', target='plain', yields=(2.0,), prices=(3.0,), costs=(1.0,), rates=(0.1,))
        >>> bid.rates
        (0.1,)
    """

    law_classes = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        yields: Sequence[float],
        prices: Sequence[float],
        costs: Sequence[float],
        rates: Sequence[float],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.yields = yields
        self.prices = prices
        self.costs = costs
        self.rates = rates

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for sequences without one value per crop.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The sequences have different lengths.
        """
        super()._prepare(context)
        lengths = {len(self.yields), len(self.prices), len(self.costs), len(self.rates)}
        if len(lengths) != 1:
            error_msg = (
                f'The yields, prices, costs, and rates of {self.label!r} have the lengths {sorted(lengths)}. Give one '
                'value per crop.'
            )
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the equation of the crop of highest bid.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation of the crop of a cell.
        """
        crop, best = SR(0), SR(0)
        crops = zip(self.yields, self.prices, self.costs, self.rates, strict=True)
        for index, (yield_, price, cost, rate) in enumerate(crops, start=1):
            bid = yield_ * (price - cost) - yield_ * rate * s.target.distance
            crop, best = where(bid > best, index, crop), where(bid > best, bid, best)
        return [EquationLaw(s.target.crop == crop, target=s.target.crop)]


class VonThunenSystem(BaseSystem):
    """A square plain around one market at its centre.

    Every cell grows the crop of highest bid rent. A crop of a high yield per cell bids most near the market, and its
    bid falls fastest with distance, so the land use forms concentric rings whose boundaries lie where the bids of
    neighbouring crops are equal (Giordano 2024 §7). The system has the states `plain.crop` and `plain.distance`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.economics.resources import VonThunenSystem
        >>> trajectory = simulate(VonThunenSystem(), until=1.0)
        >>> sorted(set(trajectory.get_values('plain.crop')[0].ravel().tolist()))
        [0.0, 1.0, 2.0, 3.0]
    """

    field = 'economics/resources'
    topics = ('land use', 'bid rent', 'static', 'lattice')
    references = ('Giordano 2024, arXiv 2407.02181', 'von Thünen 1826')

    def __init__(
        self,
        size: int = 41,
        *,
        yields: Sequence[float] = (4.0, 2.0, 1.0),
        prices: Sequence[float] = (3.0, 3.0, 3.5),
        costs: Sequence[float] = (1.0, 0.5, 0.5),
        rates: Sequence[float] = (0.2, 0.2, 0.2),
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.yields = yields
        self.prices = prices
        self.costs = costs
        self.rates = rates

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        centre = (self.size - 1) / 2
        rows, columns = np.indices((self.size, self.size))
        distance = np.hypot(rows - centre, columns - centre)
        plain = FieldElement(
            'plain',
            element=LandCellElement,
            space=LatticeSpace((self.size, self.size)),
            values={'distance': distance},
        )
        bid = BidRentInteraction(
            'bid',
            target='plain',
            yields=self.yields,
            prices=self.prices,
            costs=self.costs,
            rates=self.rates,
        )
        return CompositeSystem(self.label, elements=[plain], interactions=[bid], time=StaticTime())
