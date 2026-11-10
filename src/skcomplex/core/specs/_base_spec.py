"""Define the abstract base of every spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseSpec:
    """An abstract spec, a frozen value that a class reads, records, or builds.

    A spec is a frozen dataclass whose fields are what it is. A spec declares what a class reads, records what a run
    produced, or holds what a compiler built. It never changes and holds only its fields, the hook `_check`, and readers
    of its own fields. A spec is a value, not an estimator: it has neither `prepare` nor `run`, and the class that owns
    it calls `_check()` when it reads it. A child that validates its fields overrides `_check()`, and a spec is changed
    with `dataclasses.replace`.

    Examples:
        >>> from dataclasses import dataclass
        >>> from skcomplex.core.specs import BaseSpec
        >>> @dataclass(frozen=True)
        ... class SpanSpec(BaseSpec):
        ...     start: float
        ...     end: float
        >>> SpanSpec(0.0, 1.0).end
        1.0
    """

    def _check(self) -> None:
        """Validate the fields of the spec.

        It raises a named error for an invalid field. By default, it accepts every field.
        """
