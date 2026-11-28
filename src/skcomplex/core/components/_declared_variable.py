"""Define the record of a variable a component declares from data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._parameter import _Parameter
from ._role_spec import RoleSpec
from ._state import _State


@dataclass(frozen=True)
class _DeclaredVariable:
    """A state, parameter, or role that a component takes as data.

    It holds the initial value of the variable, or the label a role binds.

    Args:
        value:
            The initial value of the variable, as the default of an annotated argument gives it.

        spec:
            The spec of the variable, as the annotation of an annotated argument gives it.
    """

    value: object
    spec: _State | _Parameter | RoleSpec
