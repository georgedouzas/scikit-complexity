"""Define the role spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ..types import ElementKind
from ._base_spec import BaseSpec
from ._parameter_spec import ParameterSpec
from ._state_spec import StateSpec


@dataclass(frozen=True)
class RoleSpec(BaseSpec):
    """Spec of a role of an interaction, in the annotation of an argument of its constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`. A variadic
    role binds a sequence of them, and a noise role binds a `WienerProcessSpec`. An optional role may be bound to
    `None`, and then binds nothing. A role also states what it needs from what fills it: the kinds of element it accepts
    and the names of the states it reads, where a name that is an argument of the constructor of the interaction means
    the value of that argument.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

        noise:
            Whether the role binds a set of Wiener processes instead of a path.

        optional:
            Whether the role may be bound to `None`, so it binds nothing.

        elements:
            The kinds of element the role accepts, with `ElementKind.ANY` for all of them. Empty accepts any element.

        states:
            The names of the states the role reads, or of the arguments of the constructor that hold these names.

    Examples:
        >>> from skcomplex.core.specs import RoleSpec
        >>> from skcomplex.core.types import ElementKind
        >>> RoleSpec(variadic=True).variadic
        True
        >>> role = RoleSpec(elements=(ElementKind.GROUP,), states=('state',))
        >>> role.accepts(ElementKind.ELEMENT), role.accepts(ElementKind.GROUP), role.states
        (False, True, ('state',))
    """

    variadic: bool = False
    noise: bool = False
    optional: bool = False
    elements: tuple[ElementKind, ...] = ()
    states: tuple[str, ...] = ()

    def _check(self) -> None:
        """Raise an error for a role that is both variadic and a noise role.

        Raises:
            ValidationError:
                The role is both variadic and a noise role.
        """
        if self.variadic and self.noise:
            error_msg = 'A role is either variadic or a noise role, not both.'
            raise ValidationError(error_msg)

    def accepts(self, kind: ElementKind) -> bool:
        """Check whether the role accepts an element of a kind.

        Args:
            kind:
                The kind of the element.

        Returns:
            accepted:
                Whether the role names the kind, names every kind, or names none.
        """
        return not self.elements or ElementKind.ANY in self.elements or kind in self.elements


type Spec = StateSpec | ParameterSpec | RoleSpec
