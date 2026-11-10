"""Define the law that replaces a whole array state by the output of a function."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw


class ArrayUpdateLaw(BaseLaw):
    """A law that replaces a whole array state by the output of a function.

    At every step the function receives the state of every member as one array, shaped like the lattice or with one row
    per member of a group on the edges or hyperedges of a network. A state of several components per member has its
    components on a last axis. The function may also receive the parameters of the element it names, and `random`, the
    random generator, `time`, the time of the step, and `space`, the space of the group, when it names them. It returns
    the new array.

    Unlike `UpdateLaw`, which gives the next value of each member by an expression, it hands the whole array to a Python
    function. Choose it when the change depends on the whole array at once, such as one random site.

    It appears in many fields. A sandpile adds a grain at a random site, a molecular model replays recorded
    configurations, and a genetic model mutates one random site of a genome.

    Args:
        target:
            The symbol of a lattice state, or the vector of the symbols of its components.

        function:
            The function `f(values, *, random, **parameters)`, which may also name `time` and `space`.

        label:
            The name of the law, or the empty string for a law without one.

        score:
            The function `score(values, **parameters)` of the array of a state on a `LatticeSpace`, which returns one
            number per row, and which a run records at every frame as `<owner>.score`, or `None` for no record.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import ArrayUpdateLaw
        >>> ArrayUpdateLaw(var('height'), lambda values, *, random: values + 1).prepare().targets_
        (height,)
    """

    law_tag: ClassVar[str] = 'update'

    def __init__(
        self,
        target: Expression | FreeModuleElement,
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
        """Return the lattice state the update changes, or the symbols of its components."""
        return tuple(self.target) if isinstance(self.target, FreeModuleElement) else (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what array updates require: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _get_law_requirements(self) -> RequirementSpec:
        """Return what this array update requires of a system: only a lattice with a score."""
        if self.score is None:
            return self._get_requirements()
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a function that cannot be called.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if not callable(self.function):
            error_msg = f'The function of an array update must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
