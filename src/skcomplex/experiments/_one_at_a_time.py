"""Define the design that moves one name at a time from its lower to its upper bound."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseDesign
from .types import Bounds


class OneAtATimeDesign(BaseDesign):
    """Design that moves every name from its lower to its upper bound with the others at their midpoints.

    Attributes:
        rows_:
            Two rows per name, at its lower and then at its upper bound, set by `prepare` on a pair `(low, high)` per
            name.

    Examples:
        >>> from skcomplex.experiments import OneAtATimeDesign
        >>> OneAtATimeDesign().prepare({'r': (0.0, 1.0), 'K': (50, 150)}).run().rows_
        [{'r': 0.0, 'K': 100.0}, {'r': 1.0, 'K': 100.0}, {'r': 0.5, 'K': 50.0}, {'r': 0.5, 'K': 150.0}]
    """

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set the rows at the lower and the upper bound of every name, with the others at their midpoints."""
        pairs = self._check_pairs(bounds)
        middle = {name: (low + high) / 2 for name, (low, high) in pairs.items()}
        self.rows_ = [{**middle, name: bound} for name, bound_pair in pairs.items() for bound in bound_pair]
