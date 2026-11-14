"""Define the short annotations of states, parameters, and roles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ._parameter_spec import ParameterSpec
from ._role_spec import RoleSpec
from ._state_spec import StateSpec

type State[T] = Annotated[T, StateSpec()]
type Parameter[T] = Annotated[T, ParameterSpec()]
type Role = Annotated[str, RoleSpec()]
