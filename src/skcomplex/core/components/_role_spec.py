"""Define the role spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..base import BaseSpec
from ..enums import ElementKind
from ..errors import ValidationError
from ._parameter_spec import _ParameterSpec
from ._state_spec import _StateSpec


@dataclass(frozen=True)
class RoleSpec(BaseSpec):
    """A spec of a role of an interaction: what it binds and which values it reads.

    It annotates an argument of an interaction, such as `prey: Annotated[str, RoleSpec(states=('n',))]`, and says what
    the argument binds: one element, one group, or one member written `<group>[<index>]`, a sequence of them when
    variadic, or Wiener processes when it is a noise role. It names the kinds of element the role accepts and the values
    it reads, so a system rejects a wrong binding before a run. An optional role may be bound to `None`. It types the
    predator and the prey of predation, the payer of a payment, and the two ends of a spring.

    Read more in the user guide, under Interactions.

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

    See Also:
        Interaction: Binds elements through its roles.

    Examples:
        >>> from skcomplex.core.components import RoleSpec
        >>> from skcomplex.core.enums import ElementKind
        >>> role = RoleSpec(elements=(ElementKind.GROUP,), states=('state',))
        >>> role.accepts(ElementKind.ELEMENT), role.accepts(ElementKind.GROUP)
        (False, True)
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


type _Spec = _StateSpec | _ParameterSpec | RoleSpec
