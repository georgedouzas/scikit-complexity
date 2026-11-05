"""Define the spec of a variable an element declares from data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._parameter_spec import ParameterSpec
from ._state_spec import StateSpec


@dataclass(frozen=True)
class DeclaredVariableSpec(BaseSpec):
    """Spec of a state or a parameter that an element takes as data, with its initial value.

    Args:
        value:
            The initial value of the variable, as the default of an annotated argument gives it.

        spec:
            The spec of the variable, as the annotation of an annotated argument gives it.

    Raises:
        ValidationError:
            The spec is neither a `StateSpec` nor a `ParameterSpec`, raised by `prepare`.

    Examples:
        >>> from skcomplex.specs import DeclaredVariableSpec, StateSpec
        >>> DeclaredVariableSpec(1.5, StateSpec(algebraic=True)).value
        1.5
    """

    value: object
    spec: StateSpec | ParameterSpec

    def _check(self) -> None:
        """Raise an error for a spec that declares neither a state nor a parameter.

        Raises:
            ValidationError:
                The spec is neither a `StateSpec` nor a `ParameterSpec`.
        """
        if not isinstance(self.spec, StateSpec | ParameterSpec):
            error_msg = f'A declared variable takes a StateSpec or a ParameterSpec, but its spec is {self.spec!r}.'
            raise ValidationError(error_msg)
        self.spec._validated()
