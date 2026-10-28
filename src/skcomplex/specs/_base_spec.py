"""Define the base of every spec: a frozen value whose fields are what it is."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Self

from ..base import BaseConfig


@dataclass(frozen=True)
class BaseSpec(BaseConfig):
    """Base of every spec: a frozen dataclass whose fields are what it is and whose `check()` validates them.

    A spec declares what a class reads, records what a run produced, or holds what a compiler built. It never changes,
    holds only its fields, `check()`, and readers of its own fields, and is prepared and run like any configurable
    object, both doing nothing. A child validates its fields in `_check()`.

    Examples:
        >>> from dataclasses import dataclass
        >>> from skcomplex.specs import BaseSpec
        >>> @dataclass(frozen=True)
        ... class SpanSpec(BaseSpec):
        ...     start: float
        ...     end: float
        >>> SpanSpec(0.0, 1.0).check().end
        1.0
    """

    def _is_checked(self) -> None:
        """Accept a spec that was not checked, since a frozen spec cannot be marked."""

    def check(self) -> Self:
        """Validate the fields of the spec.

        Returns:
            self:
                The spec.

        Raises:
            ValidationError:
                A field is invalid, named in the message.
        """
        self._check()
        return self
