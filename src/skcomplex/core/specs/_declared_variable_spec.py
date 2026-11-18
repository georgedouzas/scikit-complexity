"""Define the spec of a variable a component declares from data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._parameter_spec import ParameterSpec
from ._role_spec import RoleSpec
from ._state_spec import StateSpec


@dataclass(frozen=True)
class DeclaredVariableSpec(BaseSpec):
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

    Examples:
        >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
        >>> DeclaredVariableSpec(1.5, StateSpec(algebraic=True)).value
        1.5
    """

    value: object
    spec: StateSpec | ParameterSpec | RoleSpec

    def _check(self) -> None:
        """Raise an error for a spec that declares no state, parameter, or role.

        Raises:
            ValidationError:
                The spec is no `StateSpec`, `ParameterSpec`, or `RoleSpec`.
        """
        if not isinstance(self.spec, StateSpec | ParameterSpec | RoleSpec):
            error_msg = (
                f'A declared variable takes a StateSpec, a ParameterSpec, or a RoleSpec, but its spec is {self.spec!r}.'
            )
            raise ValidationError(error_msg)
        self.spec._check()
