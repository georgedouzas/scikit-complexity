"""Define the role spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._parameter_spec import ParameterSpec
from ._state_spec import StateSpec


@dataclass(frozen=True)
class RoleSpec(BaseSpec):
    """Spec of a role of an interaction, in the annotation of an argument of its constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`. A variadic
    role binds a sequence of them, and a noise role binds a `WienerProcessSpec`. An optional role, like a noise role,
    may stay unbound, `None`, and then binds nothing.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

        noise:
            Whether the role binds a set of Wiener processes instead of a path.

        optional:
            Whether the role may stay unbound, `None`.

    Examples:
        >>> from skcomplex.specs import RoleSpec
        >>> RoleSpec(variadic=True).check().variadic
        True
    """

    variadic: bool = False
    noise: bool = False
    optional: bool = False

    def _check(self) -> None:
        """Raise an error for a role that is both variadic and a noise role.

        Raises:
            ValidationError:
                The role is both variadic and a noise role.
        """
        if self.variadic and self.noise:
            error_msg = 'A role is either variadic or a noise role, not both.'
            raise ValidationError(error_msg)


type Spec = StateSpec | ParameterSpec | RoleSpec
