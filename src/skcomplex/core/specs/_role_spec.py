"""Define the role spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..enums import ElementKind
from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._parameter_spec import ParameterSpec
from ._state_spec import StateSpec


@dataclass(frozen=True)
class RoleSpec(BaseSpec):
    """A spec of a role of an interaction, the elements it binds and reads.

    It is given in the annotation of an argument of the constructor of an interaction. A plain role binds one element
    path, one group path, or one member of a group, written `<group>[<index>]`. A variadic role binds a sequence of
    them, and a noise role binds a `WienerProcessSpec`. An optional role may be bound to `None`, and then binds nothing.
    A role also states what it needs from what fills it: the kinds of element it accepts and the names of the states it
    reads. A name that is an argument of the constructor means the value of that argument.

    It appears in every interaction, such as the predator and the prey of predation, the source and the target of a
    payment, and the two ends of a spring.

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

        domains:
            The domain tags the role accepts for the states it reads, or for every state of the element when it names
            none, such as `('real', 'integer')`. Empty accepts any domain.

    Examples:
        >>> from skcomplex.core.specs import RoleSpec
        >>> from skcomplex.core.enums import ElementKind
        >>> RoleSpec(variadic=True).variadic
        True
        >>> role = RoleSpec(elements=(ElementKind.GROUP,), states=('state',))
        >>> role.accepts(ElementKind.ELEMENT), role.accepts(ElementKind.GROUP), role.states
        (False, True, ('state',))
        >>> RoleSpec(domains=('boolean',)).domains
        ('boolean',)
    """

    variadic: bool = False
    noise: bool = False
    optional: bool = False
    elements: tuple[ElementKind, ...] = ()
    states: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()

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
