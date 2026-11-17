"""Define the short annotation of roles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ._role_spec import RoleSpec

type Role = Annotated[str, RoleSpec()]
