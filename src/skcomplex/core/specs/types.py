"""Define the short annotation of a role."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ._role_spec import RoleSpec

type Role = Annotated[str, RoleSpec()]
