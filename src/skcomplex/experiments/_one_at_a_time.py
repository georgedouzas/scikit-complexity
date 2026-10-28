"""Define the design that moves one name at a time from its lower to its upper bound."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ..types import Bounds
from ._base import BaseDesign


class OneAtATimeDesign(BaseDesign):
    """Design that moves every name from its lower to its upper bound with the others at their midpoints.

    Attributes:
        rows_:
            Two rows per name, at its lower and then at its upper bound, set by `prepare` on a pair `(low, high)` per
            name.

    Examples:
        >>> from skcomplex.experiments import OneAtATimeDesign
        >>> OneAtATimeDesign().check().check().prepare({'r': (0.0, 1.0), 'K': (50, 150)}).run()
        [{'r': 0.0, 'K': 100.0}, {'r': 1.0, 'K': 100.0}, {'r': 0.5, 'K': 50.0}, {'r': 0.5, 'K': 150.0}]
    """

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set the rows at the lower and the upper bound of every name, with the others at their midpoints."""
        bounds = self._check_bounds(bounds, distributions=False)
        pairs = {name: tuple(map(float, bound)) for name, bound in bounds.items()}  # type: ignore[call-overload]
        middle = {name: (low + high) / 2 for name, (low, high) in pairs.items()}
        self.rows_ = [{**middle, name: bound} for name, bound_pair in pairs.items() for bound in bound_pair]
