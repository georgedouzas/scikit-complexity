"""Define the law that replaces a whole array state by the output of a function."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces import LatticeSpace, NetworkSpace
from ...components.spaces._discrete_time import _DiscreteTime
from ...errors import ValidationError


@dataclass(frozen=True)
class ArrayUpdateLaw(BaseLaw):
    """A law that replaces a whole array state by the output of a function.

    The compiler runs it for an `UpdateLaw` with `arrays=True`, whose value is a NumPy function of the whole array,
    such as `UpdateLaw(s.target.height, drop, arrays=True, stochastic=True)` with
    `def drop(values, *, random, **kwargs): ...`. At every step the function receives the state of every member as one
    array, shaped like the lattice or with one row per member of a group on the edges or hyperedges of a network. A
    state of several values per member has them on a last axis. The function also receives the keywords of every rule
    of an `UpdateLaw`: the parameters of the element, `time`, `space`, `degree`, which is `None`, and `random`, the
    generator when the law is stochastic and `None` otherwise. It topples a sandpile, flips the spins of a lattice, and
    moves the cells of a tissue.

    Read more in the user guide, under Laws.

    Args:
        target:
            The symbol of a lattice state, or the vector of the symbols of its components.

        function:
            The function `f(values, **kwargs)` of the array and the keywords of a rule.

        label:
            The name of the law, or the empty string for a law without one.

        score:
            The function `score(values, **kwargs)` of the array of a state on a `LatticeSpace` and the keywords of a
            rule, with `random` set to `None`, which returns one number per row, and which a run records at every
            frame as `<owner>.score`, or `None` for no record.

        stochastic:
            Whether the function draws numbers, so that it receives the random generator as `random`.

    See Also:
        RelaxationLaw: Relaxes a lattice until it is stable.
        UpdateLaw: Sets the new value of a state at every step.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.engine.compile._array_update_law import ArrayUpdateLaw
        >>> law = ArrayUpdateLaw(var('h'), lambda values, **kwargs: values.clip(max=3))
        >>> law.stochastic, law.score
        (False, None)
    """

    law_tag: ClassVar[str] = 'update'

    target: Expression | FreeModuleElement
    function: Callable[..., object]
    _: KW_ONLY
    label: str = ''
    score: Callable[..., object] | None = None
    stochastic: bool = False

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'UpdateLaw'

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice state the update changes, or the symbols of its components."""
        return tuple(self.target) if isinstance(self.target, FreeModuleElement) else (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what array updates require: discrete time on a lattice or a network."""
        return RequirementSpec(time=(_DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _get_law_requirements(self) -> RequirementSpec:
        """Return what this array update requires of a system: only a lattice with a score."""
        if self.score is None:
            return self._get_requirements()
        return RequirementSpec(time=(_DiscreteTime,), space=(LatticeSpace,))

    def _check(self) -> None:
        """Raise an error for a function that cannot be called."""
        super()._check()
        if not callable(self.function):
            error_msg = f'The function of an array update must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
