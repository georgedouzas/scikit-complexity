"""Define the law that sets the value a state takes after every solver step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from ..types import Target
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class _BoundaryLaw(BaseLaw):
    """A law that sets the value a state takes after every solver step.

    After each step of the solver the state takes `value`, an expression of the states, such as a reflection at a wall
    or a wrap around a periodic box. A boundary leaves the equations of a system as they are, so it combines with every
    other law in continuous or discrete time.

    Unlike `EventLaw`, which finds the exact time a guard crosses zero, it corrects the state once per step. Choose
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
    """

    law_tag: ClassVar[str] = 'update'

    target: Target
    value: object
    _: KW_ONLY
    label: str = ''

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'UpdateLaw'

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
        super()._check()
        self._check_array('value')

    def _get_value(self, inputs: object) -> object:
        """Return the new value at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            value:
                The new value.
        """
        return self._evaluate(self.value, inputs)
