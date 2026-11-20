"""Define the spec of a variable a component declares from data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..base import BaseSpec
from ..errors import ValidationError
from ._parameter_spec import _ParameterSpec
from ._role_spec import RoleSpec
from ._state_spec import _StateSpec


@dataclass(frozen=True)
class _DeclaredVariableSpec(BaseSpec):
    """A spec of a state, parameter, or role that a component takes as data.

    It holds the initial value of the variable, or the label a role binds.

    Args:
        value:
            The initial value of the variable, as the default of an annotated argument gives it.

        spec:
            The spec of the variable, as the annotation of an annotated argument gives it.

    Raises:
        ValidationError:
            The spec is no `StateSpec`, `ParameterSpec`, or `RoleSpec`, raised when its owner checks it.
    """

    value: object
    spec: _StateSpec | _ParameterSpec | RoleSpec

    def _check(self) -> None:
        """Raise an error for a spec that declares no state, parameter, or role.

        Raises:
            ValidationError:
                The spec is no `StateSpec`, `ParameterSpec`, or `RoleSpec`.
        """
        if not isinstance(self.spec, _StateSpec | _ParameterSpec | RoleSpec):
            error_msg = (
                f'A declared variable takes a StateSpec, a ParameterSpec, or a RoleSpec, but its spec is {self.spec!r}.'
            )
            raise ValidationError(error_msg)
        self.spec._check()
