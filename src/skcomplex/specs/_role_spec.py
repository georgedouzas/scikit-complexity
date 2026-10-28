"""Define the role spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Self

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._parameter_spec import ParameterSpec
from ._state_spec import StateSpec


@dataclass(frozen=True)
class RoleSpec(BaseSpec):
    """Spec of a role of an interaction, in the annotation of an argument of its constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`. A variadic
    role binds a sequence of them, and a noise role binds a `WienerProcessSpec`.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

        noise:
            Whether the role binds a set of Wiener processes instead of a path.

    Examples:
        >>> from skcomplex.specs import RoleSpec
        >>> RoleSpec(variadic=True).check().variadic
        True
    """

    variadic: bool = False
    noise: bool = False

    def _check(self) -> Self:
        """Raise an error for a role that is both variadic and a noise role.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The role is both variadic and a noise role.
        """
        if self.variadic and self.noise:
            error_msg = 'A role is either variadic or a noise role, not both.'
            raise ValidationError(error_msg)
        return self


type Spec = StateSpec | ParameterSpec | RoleSpec
