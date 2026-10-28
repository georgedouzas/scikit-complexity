"""Define the design given by its rows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ..errors import ValidationError
from ..types import Bounds
from ._base import BaseDesign


class RowsDesign(BaseDesign):
    """Design given by its rows, which prepares on no bounds.

    Args:
        rows:
            One mapping from factory argument or dotted path to value per row.

    Attributes:
        rows_:
            A copy of the rows, set by `prepare`.

    Examples:
        >>> from skcomplex.experiments import RowsDesign
        >>> RowsDesign([{'growth.r': 0.5}, {'growth.r': 1.0}]).check().check().prepare().run()
        [{'growth.r': 0.5}, {'growth.r': 1.0}]
    """

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = rows

    def _check(self) -> None:
        """Raise an error for a row that is not a mapping."""
        for row in self.rows:
            if not isinstance(row, Mapping):
                error_msg = f'Every row of a design must be a mapping from name to value, not {row!r}.'
                raise ValidationError(error_msg)

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set a copy of the given rows, raising an error for bounds, which a rows design does not read."""
        if bounds:
            error_msg = f'A rows design takes its values from its rows, not from the bounds of {sorted(bounds)}.'
            raise ValidationError(error_msg)
        self.rows_ = [dict(row) for row in self.rows]
