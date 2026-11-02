"""The choice of the crop of highest bid rent on every cell of a plain."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR

from ....core.base import BasePart
from ....core.interactions import Interaction
from ....core.laws import EquationLaw
from ....core.laws.utils import where
from ....errors import ValidationError
from ....specs import RoleSpec


class BidRentInteraction(Interaction):
    """Land use of highest bid rent, where crop `i` bids `Y_i (p_i - c_i) - Y_i f_i d` at the distance `d` of a market.

    A cell grows the crop of the highest positive bid, and lies idle as crop 0 when no bid is positive.

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
            The four sequences do not have one value per crop, raised by `check`.

    Examples:
        >>> from skcomplex.models.economics.resources import BidRentInteraction
        >>> bid = BidRentInteraction('bid', target='plain', yields=(2.0,), prices=(3.0,), costs=(1.0,), rates=(0.1,))
        >>> bid.check().rates
        (0.1,)
    """

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

    def _check(self) -> None:
        """Raise an error for sequences without one value per crop.

        Raises:
            ValidationError:
                The sequences have different lengths.
        """
        super()._check()
        lengths = {len(self.yields), len(self.prices), len(self.costs), len(self.rates)}
        if len(lengths) != 1:
            error_msg = (
                f'The yields, prices, costs, and rates of {self.label!r} have the lengths {sorted(lengths)}. Give one '
                'value per crop.'
            )
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
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
