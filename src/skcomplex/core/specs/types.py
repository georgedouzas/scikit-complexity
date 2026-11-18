"""Define the short annotations of roles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from ._role_spec import RoleSpec

type Role = Annotated[str, RoleSpec()]
type Roles = Annotated[Sequence[str], RoleSpec(variadic=True)]
