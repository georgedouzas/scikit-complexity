"""Define the domains that variables take values from."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import Enum

import numpy as np


class Domain(Enum):
    """Set of values a variable can take.

    Examples:
        >>> from skcomplex.core import Domain
        >>> Domain.POSITIVE.contains(2.0)
        True
        >>> Domain.NONNEGATIVE_INTEGER.contains(1.5)
        False
    """

    REAL = 'real'
    POSITIVE = 'positive'
    NONNEGATIVE = 'nonnegative'
    INTEGER = 'integer'
    NONNEGATIVE_INTEGER = 'nonnegative integer'

    @property
    def assumptions(self) -> dict[str, bool]:
        """SymPy assumptions of the symbols of the domain."""
        return {
            Domain.REAL: {'real': True},
            Domain.POSITIVE: {'positive': True},
            Domain.NONNEGATIVE: {'nonnegative': True},
            Domain.INTEGER: {'integer': True},
            Domain.NONNEGATIVE_INTEGER: {'integer': True, 'nonnegative': True},
        }[self]

    def contains(self, value: object) -> bool:
        """Check that every component of a value belongs to the domain.

        Args:
            value:
                A number or an array of numbers.

        Returns:
            is_contained:
                Whether every component is finite and belongs to the domain.
        """
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return False
        if not np.all(np.isfinite(array)):
            return False
        integral = bool(np.all(array == np.round(array)))
        checks = {
            Domain.REAL: True,
            Domain.POSITIVE: bool(np.all(array > 0)),
            Domain.NONNEGATIVE: bool(np.all(array >= 0)),
            Domain.INTEGER: integral,
            Domain.NONNEGATIVE_INTEGER: integral and bool(np.all(array >= 0)),
        }
        return checks[self]
