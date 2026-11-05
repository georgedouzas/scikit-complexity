"""Define drive laws, which change a whole lattice state at once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw


class DriveLaw(BaseLaw):
    """Change of a whole lattice state at once by a function of its array, such as adding 1 at a random site.

    The function receives the state as an array of the lattice shape, keyword arguments for the parameters of the
    element that it names, and, when it names them, `random`, the random generator of the simulation, and `time`, the
    time of the step, and returns the new array.

    Args:
        target:
            The symbol of a lattice state.

        function:
            The function `f(values, *, random, **parameters)`, which may also name `time`.

        label:
            The name of the law, or the empty string for a law without one.

        score:
            The function `score(values, **parameters)` of the array of a state on a `LatticeSpace`, which returns one
            number per row, and which a run records at every frame as `<owner>.score`, or `None` for no record.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import DriveLaw
        >>> DriveLaw(var('height'), lambda values, *, random: values + 1).prepare().targets_
        (height,)
    """

    def __init__(
        self,
        target: Expression,
        function: Callable[..., object],
        *,
        label: str = '',
        score: Callable[..., object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.function = function
        self.score = score

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice state the drive changes."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what drives require of a system: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _get_law_requirements(self) -> RequirementSpec:
        """Return what this drive requires of a system: only a lattice with a score."""
        if self.score is None:
            return self._get_requirements()
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace,))

    def _check(self) -> None:
        """Raise an error for a function that cannot be called."""
        if not callable(self.function):
            error_msg = f'The function of a drive must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
