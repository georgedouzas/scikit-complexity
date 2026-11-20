"""Define the array update the compiler runs for an update law of a whole array."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ...errors import ValidationError


class ArrayUpdateLaw(BaseLaw):
    """A law that replaces a whole array state by the output of a function.

    The compiler runs it for an `UpdateLaw` of a state whose value is a NumPy function of the whole array, such as
    `UpdateLaw(s.target.height, drop)` with `def drop(values, *, random): ...`. At every step the function receives the
    state of every member as one array, shaped like the lattice or with one row per member of a group on the edges or
    hyperedges of a network. A state of several components per member has its components on a last axis. The function
    may also receive the parameters of the element, and `random`, `time`, and `space` when it names them.

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
