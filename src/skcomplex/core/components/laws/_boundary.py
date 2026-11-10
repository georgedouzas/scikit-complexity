"""Define the law that sets the value a state takes after every solver step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...specs import RequirementSpec
from ..types import Target
from ._base import BaseLaw


class BoundaryLaw(BaseLaw):
    """A law that sets the value a state takes after every solver step.

    After each step of the solver the state takes `value`, an expression of the states, such as a reflection at a wall
    or a wrap around a periodic box. A boundary leaves the equations of a system as they are, so it combines with every
    other law in continuous or discrete time.

    Unlike `CrossingLaw`, which finds the exact time a guard crosses zero, it corrects the state once per step. Choose
    it when a state must stay inside a domain and the exact time of contact does not matter.

    It appears in many fields. Particles in a periodic box wrap around its edges, a random walker reflects off a wall,
    and an angle wraps into the interval from minus pi to pi.

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

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a value that is a NumPy array.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        self._check_array('value')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the new value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.value, inputs)
