"""Define boundary laws, the values states take after every solver step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...specs import RequirementSpec
from ..types import Target
from ._base import BaseLaw


class BoundaryLaw(BaseLaw):
    """Value a state takes after every solver step.

    Examples are a reflection at a wall or a wrap around a periodic box. Boundaries leave the form of a system as is, so
    they combine with every other law in any time.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)` that gives the new value.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import floor, var
        >>> from skcomplex.core.components.laws import BoundaryLaw
        >>> x = var('x')
        >>> BoundaryLaw(x, x - floor(x)).prepare().run({'x': 2.25}).value_
        0.25
    """

    law_tag: ClassVar[str] = 'update'

    def __init__(self, target: Target, value: object, *, label: str = '') -> None:
        super().__init__(label)
        self.target = target
        self.value = value

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the boundary sets."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: nothing.

        It holds in any time.
        """
        return RequirementSpec()

    def _check(self) -> None:
        """Raise an error for a value that is a NumPy array."""
        self._check_array('value')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the new value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.value, inputs)
