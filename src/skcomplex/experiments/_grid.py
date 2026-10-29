"""Define the design of every combination of given values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Iterable

from ..errors import ValidationError
from ._base import BaseDesign
from .types import Bounds


class GridDesign(BaseDesign):
    """Design of every combination of the values of every name, which prepares on a sequence of values per name.

    Attributes:
        rows_:
            One row per combination of the values, the last name varying fastest, set by `prepare`.

    Examples:
        >>> from skcomplex.experiments import GridDesign
        >>> GridDesign().check().check().prepare({'r': [0.5, 1.0], 'K': [100, 200]}).run()
        [{'r': 0.5, 'K': 100}, {'r': 0.5, 'K': 200}, {'r': 1.0, 'K': 100}, {'r': 1.0, 'K': 200}]
    """

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set one row per combination of the values, raising an error for values that are not a sequence."""
        values: dict[str, list[object]] = {}
        for name, items in (bounds or {}).items():
            if isinstance(items, str) or not isinstance(items, Iterable):
                error_msg = f'A grid design takes a sequence of values per name, but {name} has {items!r}.'
                raise ValidationError(error_msg)
            values[name] = list(items)
        names = list(values)
        self.rows_ = [dict(zip(names, combination, strict=True)) for combination in itertools.product(*values.values())]
